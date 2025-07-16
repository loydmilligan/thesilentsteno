# Data Module Documentation

## Module Overview

The Data module provides comprehensive database and data lifecycle management for The Silent Steno application. It implements a robust SQLite-based storage system with SQLAlchemy ORM, automated backup/restore functionality, schema migration management, and configurable data retention policies. The module is designed for reliability, performance, and data integrity in a meeting recording application.

## Dependencies

### External Dependencies
- `sqlalchemy` - ORM and database toolkit
- `alembic` - Database migration tool
- `sqlite3` - SQLite database driver
- `threading` - Thread management for background operations
- `logging` - Logging system
- `json` - JSON serialization
- `datetime` - Date/time operations
- `pathlib` - Path operations
- `enum` - Enumeration types
- `dataclasses` - Data structure definitions
- `typing` - Type hints
- `hashlib` - Hash functions for integrity checking
- `gzip` - Compression for backups
- `shutil` - File operations
- `uuid` - UUID generation
- `time` - Timing operations
- `contextlib` - Context managers
- `concurrent.futures` - Thread pool execution

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.config` - Configuration management
- `src.core.monitoring` - Performance monitoring

## File Documentation

### 1. `__init__.py`

**Purpose**: Package initialization and convenience functions for the complete data management system.

#### Functions

##### `initialize_database(config: dict = None) -> Database`
Initialize database with default configuration.

**Parameters:**
- `config: dict` - Optional database configuration

**Returns:**
- `Database` - Configured database instance

##### `setup_complete_data_system(config: dict = None) -> dict`
Initialize all data management components.

**Parameters:**
- `config: dict` - Optional system configuration

**Returns:**
- `dict` - Dictionary containing all initialized managers

##### `get_system_status() -> dict`
Get comprehensive system health status.

**Returns:**
- `dict` - System status information

**Usage Example:**
```python
# Initialize complete data system
system = setup_complete_data_system({
    "database": {
        "path": "data/silentst.db",
        "pool_size": 10,
        "timeout": 30
    },
    "backup": {
        "enabled": True,
        "interval": 3600,
        "retention_days": 30
    },
    "retention": {
        "session_retention_days": 90,
        "transcript_retention_days": 180
    }
})

# Get system status
status = get_system_status()
print(f"Database status: {status['database']['status']}")
print(f"Backup status: {status['backup']['last_backup']}")
print(f"Storage usage: {status['storage']['usage_percent']:.1f}%")
```

### 2. `database.py`

**Purpose**: Core database connectivity and session management with SQLAlchemy ORM.

#### Classes

##### `DatabaseConfig`
Database configuration settings.

**Attributes:**
- `database_path: str` - Path to SQLite database file
- `pool_size: int` - Connection pool size
- `timeout: float` - Connection timeout in seconds
- `enable_wal: bool` - Enable WAL mode
- `enable_foreign_keys: bool` - Enable foreign key constraints
- `journal_mode: str` - SQLite journal mode
- `synchronous: str` - Synchronous mode setting
- `cache_size: int` - Cache size in KB
- `mmap_size: int` - Memory map size

##### `Database`
Main database connection manager.

**Methods:**
- `__init__(config: DatabaseConfig)` - Initialize with configuration
- `connect()` - Establish database connection
- `disconnect()` - Close database connection
- `get_session()` - Get database session
- `create_tables()` - Create database tables
- `drop_tables()` - Drop database tables
- `execute_query(query: str, params: dict = None)` - Execute SQL query
- `get_connection_info()` - Get connection information
- `health_check()` - Perform health check
- `get_metrics()` - Get performance metrics

##### `SessionManager`
High-level session management with retry logic.

**Methods:**
- `__init__(database: Database)` - Initialize with database
- `create_session(auto_commit: bool = True)` - Create session context manager
- `execute_transaction(func: callable, *args, **kwargs)` - Execute transaction
- `bulk_insert(model_class, data_list: List[dict])` - Bulk insert data
- `bulk_update(model_class, updates: List[dict])` - Bulk update data
- `get_session_stats()` - Get session statistics

**Usage Example:**
```python
# Create database configuration
config = DatabaseConfig(
    database_path="data/silentst.db",
    pool_size=10,
    timeout=30.0,
    enable_wal=True,
    enable_foreign_keys=True
)

# Create database instance
db = Database(config)
db.connect()
db.create_tables()

# Create session manager
session_manager = SessionManager(db)

# Use session context manager
with session_manager.create_session() as session:
    # Query data
    sessions = session.query(Session).filter(
        Session.status == "completed"
    ).all()
    
    # Create new session
    new_session = Session(
        title="Team Meeting",
        start_time=datetime.now(),
        status="active"
    )
    session.add(new_session)
    session.commit()

# Execute transaction
def create_session_with_transcript(session_data, transcript_data):
    session = Session(**session_data)
    transcript = TranscriptEntry(**transcript_data)
    transcript.session = session
    return session

result = session_manager.execute_transaction(
    create_session_with_transcript,
    {"title": "Meeting", "start_time": datetime.now()},
    {"text": "Hello everyone", "speaker": "Alice"}
)
```

### 3. `models.py`

**Purpose**: Complete SQLAlchemy ORM models defining the database schema.

#### Enums

##### `SessionStatus`
Session status enumeration.
- `ACTIVE` - Session is currently recording
- `PAUSED` - Session is paused
- `COMPLETED` - Session completed successfully
- `FAILED` - Session failed
- `CANCELLED` - Session was cancelled

##### `TranscriptEntryType`
Transcript entry types.
- `SPEECH` - Speech content
- `SILENCE` - Silence period
- `NOISE` - Background noise
- `MUSIC` - Music or audio

##### `AnalysisType`
Analysis result types.
- `SUMMARY` - Meeting summary
- `ACTION_ITEMS` - Action items
- `TOPICS` - Key topics
- `SENTIMENT` - Sentiment analysis

#### Models

##### `Session`
Meeting recording session.

**Attributes:**
- `id: int` - Primary key
- `title: str` - Session title
- `start_time: datetime` - Start timestamp
- `end_time: datetime` - End timestamp
- `duration: float` - Duration in seconds
- `status: SessionStatus` - Session status
- `participant_count: int` - Number of participants
- `audio_file_path: str` - Path to audio file
- `transcript_file_path: str` - Path to transcript file
- `metadata: dict` - Additional metadata (JSON)
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update timestamp

**Methods:**
- `get_duration()` - Calculate session duration
- `add_participant(participant)` - Add participant
- `get_transcript_entries()` - Get transcript entries
- `to_dict()` - Convert to dictionary
- `is_active()` - Check if session is active

##### `TranscriptEntry`
Individual transcript segment.

**Attributes:**
- `id: int` - Primary key
- `session_id: int` - Foreign key to Session
- `text: str` - Transcript text
- `speaker: str` - Speaker name/identifier
- `start_time: float` - Start time in seconds
- `end_time: float` - End time in seconds
- `confidence: float` - Transcription confidence (0-1)
- `entry_type: TranscriptEntryType` - Entry type
- `language: str` - Language code
- `word_count: int` - Number of words
- `created_at: datetime` - Creation timestamp

**Methods:**
- `get_duration()` - Get entry duration
- `get_words_per_minute()` - Calculate WPM
- `to_dict()` - Convert to dictionary
- `merge_with(other_entry)` - Merge with another entry

##### `AnalysisResult`
AI analysis results.

**Attributes:**
- `id: int` - Primary key
- `session_id: int` - Foreign key to Session
- `analysis_type: AnalysisType` - Type of analysis
- `result_data: dict` - Analysis results (JSON)
- `confidence: float` - Analysis confidence
- `processing_time: float` - Processing time in seconds
- `model_version: str` - AI model version
- `created_at: datetime` - Creation timestamp

**Methods:**
- `get_summary()` - Get analysis summary
- `get_action_items()` - Get action items
- `get_topics()` - Get key topics
- `to_dict()` - Convert to dictionary

##### `Participant`
Session participant information.

**Attributes:**
- `id: int` - Primary key
- `session_id: int` - Foreign key to Session
- `name: str` - Participant name
- `speaker_id: str` - Speaker identifier
- `speaking_time: float` - Total speaking time
- `word_count: int` - Total word count
- `interruptions: int` - Number of interruptions
- `questions_asked: int` - Questions asked
- `sentiment_score: float` - Average sentiment
- `engagement_level: float` - Engagement level (0-1)

**Methods:**
- `get_speaking_percentage()` - Get speaking time percentage
- `get_words_per_minute()` - Calculate average WPM
- `to_dict()` - Convert to dictionary

##### `User`
System user configuration.

**Attributes:**
- `id: int` - Primary key
- `username: str` - Username
- `email: str` - Email address
- `preferences: dict` - User preferences (JSON)
- `created_at: datetime` - Creation timestamp
- `last_login: datetime` - Last login timestamp

##### `Configuration`
System configuration settings.

**Attributes:**
- `id: int` - Primary key
- `key: str` - Configuration key
- `value: str` - Configuration value
- `category: str` - Configuration category
- `description: str` - Description
- `updated_at: datetime` - Last update timestamp

##### `FileInfo`
File metadata and storage information.

**Attributes:**
- `id: int` - Primary key
- `session_id: int` - Foreign key to Session
- `file_path: str` - File path
- `file_type: str` - File type
- `file_size: int` - File size in bytes
- `checksum: str` - File checksum
- `created_at: datetime` - Creation timestamp

##### `SystemMetrics`
System performance metrics.

**Attributes:**
- `id: int` - Primary key
- `metric_name: str` - Metric name
- `metric_value: float` - Metric value
- `metric_unit: str` - Unit of measurement
- `timestamp: datetime` - Measurement timestamp
- `metadata: dict` - Additional metadata (JSON)

**Usage Example:**
```python
# Create new session
session = Session(
    title="Weekly Team Meeting",
    start_time=datetime.now(),
    status=SessionStatus.ACTIVE,
    participant_count=5
)

# Add transcript entries
transcript1 = TranscriptEntry(
    session=session,
    text="Good morning everyone, let's start the meeting.",
    speaker="Alice",
    start_time=0.0,
    end_time=3.5,
    confidence=0.95,
    entry_type=TranscriptEntryType.SPEECH
)

transcript2 = TranscriptEntry(
    session=session,
    text="Thanks Alice. I have the agenda ready.",
    speaker="Bob",
    start_time=3.5,
    end_time=6.8,
    confidence=0.92,
    entry_type=TranscriptEntryType.SPEECH
)

# Add participants
participant1 = Participant(
    session=session,
    name="Alice Johnson",
    speaker_id="SPEAKER_01",
    speaking_time=120.5,
    word_count=250,
    engagement_level=0.8
)

# Add analysis results
analysis = AnalysisResult(
    session=session,
    analysis_type=AnalysisType.SUMMARY,
    result_data={
        "summary": "Team discussed project timeline and deliverables.",
        "key_points": ["Timeline review", "Resource allocation", "Next steps"]
    },
    confidence=0.87,
    processing_time=2.3
)

# Save to database
with session_manager.create_session() as db_session:
    db_session.add_all([session, transcript1, transcript2, participant1, analysis])
    db_session.commit()

# Query with relationships
sessions = db_session.query(Session).options(
    joinedload(Session.transcript_entries),
    joinedload(Session.participants),
    joinedload(Session.analysis_results)
).filter(Session.status == SessionStatus.COMPLETED).all()
```

### 4. `migrations.py`

**Purpose**: Database schema migration management using Alembic.

#### Classes

##### `MigrationInfo`
Migration metadata information.

**Attributes:**
- `migration_id: str` - Migration identifier
- `description: str` - Migration description
- `applied_at: datetime` - Application timestamp
- `rollback_available: bool` - Rollback availability
- `checksum: str` - Migration checksum

##### `MigrationManager`
Complete migration lifecycle management.

**Methods:**
- `__init__(database: Database, config: dict = None)` - Initialize with database
- `initialize_alembic()` - Initialize Alembic configuration
- `generate_migration(message: str)` - Generate new migration
- `apply_migrations()` - Apply pending migrations
- `rollback_migration(steps: int = 1)` - Rollback migrations
- `get_migration_history()` - Get migration history
- `validate_migration(migration_id: str)` - Validate migration
- `backup_before_migration()` - Create backup before migration
- `get_pending_migrations()` - Get pending migrations

**Usage Example:**
```python
# Create migration manager
migration_manager = MigrationManager(database)

# Initialize Alembic
migration_manager.initialize_alembic()

# Generate migration for new table
migration_manager.generate_migration("Add user preferences table")

# Get pending migrations
pending = migration_manager.get_pending_migrations()
print(f"Pending migrations: {[m.description for m in pending]}")

# Apply migrations
result = migration_manager.apply_migrations()
if result.success:
    print(f"Applied {len(result.applied_migrations)} migrations")
else:
    print(f"Migration failed: {result.error}")

# Get migration history
history = migration_manager.get_migration_history()
for migration in history:
    print(f"{migration.migration_id}: {migration.description}")
    print(f"  Applied: {migration.applied_at}")
```

### 5. `backup_manager.py`

**Purpose**: Comprehensive database backup and restore functionality.

#### Classes

##### `BackupConfig`
Backup configuration settings.

**Attributes:**
- `backup_directory: str` - Backup directory path
- `backup_interval: int` - Backup interval in seconds
- `max_backups: int` - Maximum number of backups to keep
- `compress_backups: bool` - Enable compression
- `verify_backups: bool` - Verify backup integrity
- `notification_enabled: bool` - Enable notifications

##### `BackupInfo`
Backup metadata information.

**Attributes:**
- `backup_id: str` - Backup identifier
- `file_path: str` - Backup file path
- `file_size: int` - Backup file size
- `compressed_size: int` - Compressed size
- `checksum: str` - Backup checksum
- `created_at: datetime` - Creation timestamp
- `compression_ratio: float` - Compression ratio
- `verified: bool` - Verification status

##### `BackupManager`
Complete backup lifecycle management.

**Methods:**
- `__init__(database: Database, config: BackupConfig)` - Initialize with database and config
- `create_backup(description: str = None)` - Create database backup
- `restore_backup(backup_id: str)` - Restore from backup
- `verify_backup(backup_id: str)` - Verify backup integrity
- `list_backups()` - List available backups
- `delete_backup(backup_id: str)` - Delete backup
- `start_scheduled_backups()` - Start scheduled backup process
- `stop_scheduled_backups()` - Stop scheduled backups
- `cleanup_old_backups()` - Clean up old backups
- `get_backup_stats()` - Get backup statistics

**Usage Example:**
```python
# Create backup configuration
backup_config = BackupConfig(
    backup_directory="backups",
    backup_interval=3600,  # 1 hour
    max_backups=24,        # Keep 24 backups
    compress_backups=True,
    verify_backups=True
)

# Create backup manager
backup_manager = BackupManager(database, backup_config)

# Create immediate backup
backup_info = backup_manager.create_backup("Pre-migration backup")
print(f"Backup created: {backup_info.backup_id}")
print(f"File size: {backup_info.file_size / 1024 / 1024:.1f} MB")
print(f"Compression ratio: {backup_info.compression_ratio:.2f}")

# Start scheduled backups
backup_manager.start_scheduled_backups()

# List available backups
backups = backup_manager.list_backups()
for backup in backups:
    print(f"{backup.backup_id}: {backup.created_at}")
    print(f"  Size: {backup.file_size / 1024 / 1024:.1f} MB")
    print(f"  Verified: {backup.verified}")

# Restore from backup
restore_result = backup_manager.restore_backup(backup_info.backup_id)
if restore_result.success:
    print("Restore completed successfully")
else:
    print(f"Restore failed: {restore_result.error}")

# Cleanup old backups
cleanup_result = backup_manager.cleanup_old_backups()
print(f"Cleaned up {cleanup_result.deleted_count} old backups")
```

### 6. `retention_manager.py`

**Purpose**: Configurable data retention policies for storage optimization.

#### Classes

##### `RetentionRule`
Individual retention rule definition.

**Attributes:**
- `rule_id: str` - Rule identifier
- `name: str` - Rule name
- `description: str` - Rule description
- `model_class: type` - Target model class
- `retention_days: int` - Retention period in days
- `max_count: int` - Maximum count to keep
- `max_size_mb: int` - Maximum size in MB
- `preserve_condition: callable` - Condition to preserve data
- `enabled: bool` - Rule enabled status

##### `RetentionConfig`
System-wide retention configuration.

**Attributes:**
- `enabled: bool` - Enable retention management
- `cleanup_interval: int` - Cleanup interval in seconds
- `dry_run: bool` - Enable dry run mode
- `backup_before_cleanup: bool` - Create backup before cleanup
- `notification_enabled: bool` - Enable notifications
- `rules: List[RetentionRule]` - List of retention rules

##### `CleanupResult`
Cleanup operation results.

**Attributes:**
- `rule_id: str` - Rule that was applied
- `deleted_count: int` - Number of items deleted
- `freed_space: int` - Space freed in bytes
- `processing_time: float` - Processing time in seconds
- `success: bool` - Operation success status
- `error: str` - Error message if failed

##### `RetentionManager`
Policy application and scheduling.

**Methods:**
- `__init__(database: Database, config: RetentionConfig)` - Initialize with database and config
- `add_rule(rule: RetentionRule)` - Add retention rule
- `remove_rule(rule_id: str)` - Remove retention rule
- `apply_retention_rules()` - Apply all retention rules
- `apply_rule(rule_id: str)` - Apply specific rule
- `estimate_cleanup(rule_id: str = None)` - Estimate cleanup impact
- `start_scheduled_cleanup()` - Start scheduled cleanup
- `stop_scheduled_cleanup()` - Stop scheduled cleanup
- `get_retention_stats()` - Get retention statistics
- `get_storage_usage()` - Get storage usage information

**Usage Example:**
```python
# Create retention rules
session_rule = RetentionRule(
    rule_id="session_retention",
    name="Session Retention",
    description="Keep sessions for 90 days",
    model_class=Session,
    retention_days=90,
    preserve_condition=lambda session: session.status == SessionStatus.ACTIVE
)

transcript_rule = RetentionRule(
    rule_id="transcript_retention",
    name="Transcript Retention",
    description="Keep transcripts for 180 days",
    model_class=TranscriptEntry,
    retention_days=180,
    max_count=10000
)

# Create retention configuration
retention_config = RetentionConfig(
    enabled=True,
    cleanup_interval=86400,  # Daily cleanup
    dry_run=False,
    backup_before_cleanup=True,
    rules=[session_rule, transcript_rule]
)

# Create retention manager
retention_manager = RetentionManager(database, retention_config)

# Add custom rule
custom_rule = RetentionRule(
    rule_id="large_files",
    name="Large File Cleanup",
    description="Remove large files older than 30 days",
    model_class=FileInfo,
    retention_days=30,
    preserve_condition=lambda file: file.file_size < 100 * 1024 * 1024  # Keep files < 100MB
)

retention_manager.add_rule(custom_rule)

# Estimate cleanup impact
estimate = retention_manager.estimate_cleanup()
print(f"Estimated cleanup:")
print(f"  Items to delete: {estimate.total_items}")
print(f"  Space to free: {estimate.total_space / 1024 / 1024:.1f} MB")

# Apply retention rules
results = retention_manager.apply_retention_rules()
for result in results:
    print(f"Rule {result.rule_id}:")
    print(f"  Deleted: {result.deleted_count} items")
    print(f"  Freed: {result.freed_space / 1024 / 1024:.1f} MB")
    print(f"  Time: {result.processing_time:.2f}s")

# Start scheduled cleanup
retention_manager.start_scheduled_cleanup()

# Get retention statistics
stats = retention_manager.get_retention_stats()
print(f"Retention Statistics:")
print(f"  Rules active: {stats.active_rules}")
print(f"  Last cleanup: {stats.last_cleanup}")
print(f"  Total items managed: {stats.total_items}")
```

## Module Integration

The Data module integrates with other Silent Steno components:

1. **Core Events**: Publishes data events and status updates
2. **Recording Module**: Stores session and transcript data
3. **AI Module**: Stores analysis results and metrics
4. **Export Module**: Provides data for export operations
5. **System Module**: Monitors storage and performance

## Common Usage Patterns

### Complete Data System Setup
```python
# Initialize complete data system
config = {
    "database": {
        "path": "data/silentst.db",
        "pool_size": 10,
        "enable_wal": True,
        "timeout": 30
    },
    "backup": {
        "enabled": True,
        "interval": 3600,
        "directory": "backups",
        "max_backups": 24,
        "compress": True
    },
    "retention": {
        "enabled": True,
        "cleanup_interval": 86400,
        "session_retention_days": 90,
        "transcript_retention_days": 180
    },
    "migrations": {
        "auto_apply": True,
        "backup_before_migration": True
    }
}

# Setup system
system = setup_complete_data_system(config)
db = system['database']
backup_manager = system['backup']
retention_manager = system['retention']
migration_manager = system['migration']

# Check system status
status = get_system_status()
print(f"System Status: {status['overall_status']}")
```

### Session Management Workflow
```python
# Create session manager
session_manager = SessionManager(database)

# Create new session with transaction
def create_meeting_session(title, participants):
    with session_manager.create_session() as session:
        # Create session record
        meeting_session = Session(
            title=title,
            start_time=datetime.now(),
            status=SessionStatus.ACTIVE,
            participant_count=len(participants)
        )
        session.add(meeting_session)
        session.flush()  # Get ID
        
        # Add participants
        for participant_name in participants:
            participant = Participant(
                session=meeting_session,
                name=participant_name,
                speaker_id=f"SPEAKER_{len(participants)}",
                speaking_time=0.0,
                word_count=0
            )
            session.add(participant)
        
        session.commit()
        return meeting_session.id

# Use transaction
session_id = session_manager.execute_transaction(
    create_meeting_session,
    "Weekly Team Meeting",
    ["Alice", "Bob", "Charlie"]
)

# Add transcript entries
def add_transcript_entry(session_id, text, speaker, start_time, end_time):
    with session_manager.create_session() as session:
        entry = TranscriptEntry(
            session_id=session_id,
            text=text,
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            confidence=0.9,
            entry_type=TranscriptEntryType.SPEECH
        )
        session.add(entry)
        session.commit()
        return entry.id

# Add entries
add_transcript_entry(session_id, "Good morning everyone", "Alice", 0.0, 2.5)
add_transcript_entry(session_id, "Hi Alice, ready to start", "Bob", 2.5, 5.0)
```

### Backup and Recovery Workflow
```python
# Create backup before major operation
backup_info = backup_manager.create_backup("Pre-migration backup")

# Perform operation (e.g., migration)
try:
    migration_manager.apply_migrations()
    print("Migration completed successfully")
except Exception as e:
    print(f"Migration failed: {e}")
    
    # Restore from backup
    restore_result = backup_manager.restore_backup(backup_info.backup_id)
    if restore_result.success:
        print("Restored from backup successfully")
    else:
        print(f"Restore failed: {restore_result.error}")

# Scheduled backup management
backup_manager.start_scheduled_backups()

# Monitor backup status
def check_backup_status():
    stats = backup_manager.get_backup_stats()
    print(f"Backup Status:")
    print(f"  Last backup: {stats.last_backup}")
    print(f"  Total backups: {stats.total_backups}")
    print(f"  Total size: {stats.total_size / 1024 / 1024:.1f} MB")
    
    # Cleanup if needed
    if stats.total_backups > backup_config.max_backups:
        cleanup_result = backup_manager.cleanup_old_backups()
        print(f"Cleaned up {cleanup_result.deleted_count} old backups")

# Run periodic check
check_backup_status()
```

### Data Retention Management
```python
# Setup retention rules
retention_config = RetentionConfig(
    enabled=True,
    cleanup_interval=86400,  # Daily
    backup_before_cleanup=True
)

retention_manager = RetentionManager(database, retention_config)

# Add retention rules
session_rule = RetentionRule(
    rule_id="session_cleanup",
    name="Session Cleanup",
    model_class=Session,
    retention_days=90,
    preserve_condition=lambda s: s.status == SessionStatus.ACTIVE
)

transcript_rule = RetentionRule(
    rule_id="transcript_cleanup",
    name="Transcript Cleanup",
    model_class=TranscriptEntry,
    retention_days=180,
    max_count=50000
)

retention_manager.add_rule(session_rule)
retention_manager.add_rule(transcript_rule)

# Estimate cleanup impact
estimate = retention_manager.estimate_cleanup()
print(f"Cleanup Estimate:")
print(f"  Sessions to delete: {estimate.session_count}")
print(f"  Transcripts to delete: {estimate.transcript_count}")
print(f"  Space to free: {estimate.total_space / 1024 / 1024:.1f} MB")

# Apply retention if estimate is acceptable
if estimate.total_space > 100 * 1024 * 1024:  # > 100MB
    results = retention_manager.apply_retention_rules()
    for result in results:
        if result.success:
            print(f"Rule {result.rule_id}: freed {result.freed_space / 1024 / 1024:.1f} MB")
```

### Performance Monitoring
```python
# Monitor database performance
def monitor_database_performance():
    metrics = database.get_metrics()
    print(f"Database Performance:")
    print(f"  Active connections: {metrics.active_connections}")
    print(f"  Query count: {metrics.query_count}")
    print(f"  Average query time: {metrics.avg_query_time:.2f}ms")
    print(f"  Database size: {metrics.database_size / 1024 / 1024:.1f} MB")
    
    # Check for issues
    if metrics.avg_query_time > 1000:  # > 1 second
        print("WARNING: Slow query performance detected")
    
    if metrics.active_connections > 8:
        print("WARNING: High connection count")

# Monitor storage usage
def monitor_storage():
    storage_info = retention_manager.get_storage_usage()
    usage_percent = (storage_info.used_space / storage_info.total_space) * 100
    
    print(f"Storage Usage: {usage_percent:.1f}%")
    print(f"  Used: {storage_info.used_space / 1024 / 1024:.1f} MB")
    print(f"  Available: {storage_info.available_space / 1024 / 1024:.1f} MB")
    
    # Trigger cleanup if needed
    if usage_percent > 85:
        print("WARNING: High storage usage - triggering cleanup")
        retention_manager.apply_retention_rules()

# Run monitoring
monitor_database_performance()
monitor_storage()
```

### Error Handling and Recovery
```python
# Robust data operations with error handling
class DataOperationManager:
    def __init__(self, database, backup_manager, retention_manager):
        self.database = database
        self.backup_manager = backup_manager
        self.retention_manager = retention_manager
        self.session_manager = SessionManager(database)
    
    def safe_data_operation(self, operation_func, *args, **kwargs):
        """Perform data operation with backup and recovery."""
        # Create backup before operation
        backup_info = self.backup_manager.create_backup(
            f"Pre-operation backup: {operation_func.__name__}"
        )
        
        try:
            # Perform operation
            result = operation_func(*args, **kwargs)
            
            # Verify database integrity
            if not self.database.health_check():
                raise RuntimeError("Database integrity check failed")
            
            return result
            
        except Exception as e:
            print(f"Operation failed: {e}")
            
            # Restore from backup
            restore_result = self.backup_manager.restore_backup(backup_info.backup_id)
            if restore_result.success:
                print("Restored from backup successfully")
            else:
                print(f"Restore also failed: {restore_result.error}")
                raise RuntimeError("Critical data failure - manual intervention required")
            
            raise  # Re-raise original exception
    
    def cleanup_with_safety(self):
        """Perform cleanup with safety checks."""
        # Check available space
        storage_info = self.retention_manager.get_storage_usage()
        if storage_info.available_space < 1024 * 1024 * 1024:  # < 1GB
            # Estimate cleanup impact
            estimate = self.retention_manager.estimate_cleanup()
            
            if estimate.total_space > 100 * 1024 * 1024:  # > 100MB to free
                # Perform cleanup
                results = self.retention_manager.apply_retention_rules()
                
                total_freed = sum(r.freed_space for r in results if r.success)
                print(f"Cleanup freed {total_freed / 1024 / 1024:.1f} MB")
                
                return total_freed
        
        return 0

# Use robust operations
data_manager = DataOperationManager(database, backup_manager, retention_manager)

# Perform safe operations
try:
    result = data_manager.safe_data_operation(
        migration_manager.apply_migrations
    )
    print("Migration completed safely")
except Exception as e:
    print(f"Migration failed with recovery: {e}")

# Perform safe cleanup
freed_space = data_manager.cleanup_with_safety()
print(f"Cleanup completed: {freed_space / 1024 / 1024:.1f} MB freed")
```

This comprehensive documentation provides complete technical details and practical usage examples for all components in the Data module, enabling developers to implement robust data management for The Silent Steno system with proper backup, retention, and recovery capabilities.