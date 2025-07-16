-- Database Schema for The Silent Steno
-- Bluetooth AI Meeting Recorder Database Definition
-- 
-- This file defines the complete database schema for session metadata,
-- transcripts, analysis results, participants, and system configuration.
-- 
-- Database: SQLite with WAL mode
-- ORM: SQLAlchemy
-- Features: Foreign keys, indexes, constraints, triggers

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Performance optimizations
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;

-- ============================================================================
-- MAIN SESSION TABLE
-- ============================================================================

CREATE TABLE sessions (
    -- Primary key and identification
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    
    -- Session metadata
    title TEXT NOT NULL DEFAULT 'Untitled Session',
    description TEXT,
    status TEXT NOT NULL DEFAULT 'idle' CHECK (status IN ('idle', 'starting', 'recording', 'paused', 'stopping', 'processing', 'completed', 'error', 'archived')),
    
    -- Timing information
    start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER DEFAULT 0 CHECK (duration_seconds >= 0),
    
    -- Audio information
    audio_file_path TEXT,
    audio_format TEXT DEFAULT 'wav',
    audio_size_bytes INTEGER DEFAULT 0,
    audio_quality TEXT DEFAULT 'high',
    sample_rate INTEGER DEFAULT 44100,
    
    -- Participants and environment
    participant_count INTEGER DEFAULT 0 CHECK (participant_count >= 0),
    location TEXT,
    meeting_platform TEXT,
    
    -- Processing information
    transcription_completed BOOLEAN DEFAULT FALSE,
    analysis_completed BOOLEAN DEFAULT FALSE,
    processing_time_seconds INTEGER DEFAULT 0,
    
    -- Quality metrics
    transcription_confidence REAL DEFAULT 0.0 CHECK (transcription_confidence >= 0 AND transcription_confidence <= 1),
    audio_level_avg REAL DEFAULT 0.0,
    audio_level_max REAL DEFAULT 0.0,
    
    -- Metadata and configuration (JSON)
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    configuration TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Session indexes for performance
CREATE INDEX idx_session_status ON sessions(status);
CREATE INDEX idx_session_created ON sessions(created_at);
CREATE INDEX idx_session_start_time ON sessions(start_time);
CREATE INDEX idx_session_uuid ON sessions(uuid);
CREATE INDEX idx_session_title ON sessions(title);

-- Update trigger for sessions
CREATE TRIGGER update_session_timestamp 
    AFTER UPDATE ON sessions
    FOR EACH ROW
BEGIN
    UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- TRANSCRIPT ENTRIES TABLE
-- ============================================================================

CREATE TABLE transcript_entries (
    -- Primary key and relationships
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Transcript content
    text TEXT NOT NULL,
    speaker_id TEXT,
    speaker_name TEXT,
    
    -- Timing information
    start_time_seconds REAL NOT NULL CHECK (start_time_seconds >= 0),
    end_time_seconds REAL NOT NULL CHECK (end_time_seconds >= start_time_seconds),
    duration_seconds REAL NOT NULL,
    
    -- Quality metrics
    confidence REAL DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    confidence_level TEXT DEFAULT 'medium' CHECK (confidence_level IN ('low', 'medium', 'high', 'very_high')),
    
    -- Audio characteristics
    audio_level REAL DEFAULT 0.0,
    word_count INTEGER DEFAULT 0,
    speaking_rate REAL DEFAULT 0.0, -- words per minute
    
    -- Processing information
    language TEXT DEFAULT 'en',
    processing_model TEXT DEFAULT 'whisper-base',
    
    -- Metadata (JSON)
    metadata TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Transcript indexes for performance
CREATE INDEX idx_transcript_session ON transcript_entries(session_id);
CREATE INDEX idx_transcript_speaker ON transcript_entries(speaker_id);
CREATE INDEX idx_transcript_time ON transcript_entries(start_time_seconds, end_time_seconds);
CREATE INDEX idx_transcript_created ON transcript_entries(created_at);

-- ============================================================================
-- ANALYSIS RESULTS TABLE
-- ============================================================================

CREATE TABLE analysis_results (
    -- Primary key and relationships
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Analysis information
    analysis_type TEXT NOT NULL CHECK (analysis_type IN ('summary', 'action_items', 'topics', 'sentiment', 'participants', 'insights', 'comprehensive')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    
    -- Structured data (JSON)
    structured_data TEXT DEFAULT '{}',
    action_items TEXT DEFAULT '[]',
    key_topics TEXT DEFAULT '[]',
    
    -- Quality metrics
    confidence_score REAL DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_time_seconds REAL DEFAULT 0.0,
    
    -- Processing information
    model_used TEXT,
    model_version TEXT,
    prompt_template TEXT,
    
    -- Metadata (JSON)
    metadata TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Analysis indexes for performance
CREATE INDEX idx_analysis_session ON analysis_results(session_id);
CREATE INDEX idx_analysis_type ON analysis_results(analysis_type);
CREATE INDEX idx_analysis_created ON analysis_results(created_at);

-- ============================================================================
-- PARTICIPANTS TABLE
-- ============================================================================

CREATE TABLE participants (
    -- Primary key and relationships
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Participant identification
    speaker_id TEXT NOT NULL,
    name TEXT,
    role TEXT,
    organization TEXT,
    
    -- Speaking statistics
    total_speaking_time_seconds REAL DEFAULT 0.0 CHECK (total_speaking_time_seconds >= 0),
    speaking_percentage REAL DEFAULT 0.0 CHECK (speaking_percentage >= 0 AND speaking_percentage <= 100),
    interruption_count INTEGER DEFAULT 0,
    questions_asked INTEGER DEFAULT 0,
    
    -- Audio characteristics
    average_audio_level REAL DEFAULT 0.0,
    speaking_rate_wpm REAL DEFAULT 0.0, -- words per minute
    
    -- Engagement metrics
    engagement_score REAL DEFAULT 0.0,
    contribution_score REAL DEFAULT 0.0,
    
    -- Metadata (JSON)
    metadata TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(session_id, speaker_id)
);

-- Participants indexes for performance
CREATE INDEX idx_participant_session ON participants(session_id);
CREATE INDEX idx_participant_speaker ON participants(speaker_id);

-- Update trigger for participants
CREATE TRIGGER update_participant_timestamp 
    AFTER UPDATE ON participants
    FOR EACH ROW
BEGIN
    UPDATE participants SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE users (
    -- Primary key and identification
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    full_name TEXT,
    
    -- Preferences
    preferred_language TEXT DEFAULT 'en',
    timezone TEXT DEFAULT 'UTC',
    theme TEXT DEFAULT 'dark',
    
    -- Settings
    auto_transcribe BOOLEAN DEFAULT TRUE,
    auto_analyze BOOLEAN DEFAULT TRUE,
    backup_enabled BOOLEAN DEFAULT TRUE,
    notification_enabled BOOLEAN DEFAULT TRUE,
    
    -- Preferences and metadata (JSON)
    preferences TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Users indexes for performance
CREATE INDEX idx_user_username ON users(username);
CREATE INDEX idx_user_email ON users(email);

-- Update trigger for users
CREATE TRIGGER update_user_timestamp 
    AFTER UPDATE ON users
    FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE configurations (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Configuration identification
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    
    -- Metadata
    description TEXT,
    data_type TEXT DEFAULT 'string' CHECK (data_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    is_system BOOLEAN DEFAULT FALSE,
    is_user_editable BOOLEAN DEFAULT TRUE,
    
    -- Validation (JSON)
    validation_rules TEXT DEFAULT '{}',
    default_value TEXT,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(category, key)
);

-- Configuration indexes for performance
CREATE INDEX idx_config_category ON configurations(category);
CREATE INDEX idx_config_key ON configurations(key);

-- Update trigger for configurations
CREATE TRIGGER update_configuration_timestamp 
    AFTER UPDATE ON configurations
    FOR EACH ROW
BEGIN
    UPDATE configurations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- FILE INFO TABLE
-- ============================================================================

CREATE TABLE file_info (
    -- Primary key and relationships
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- File information
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL, -- audio, transcript, analysis, export
    file_format TEXT NOT NULL, -- wav, mp3, txt, pdf, etc.
    
    -- File properties
    size_bytes INTEGER NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    checksum TEXT, -- SHA-256 hash
    mime_type TEXT,
    
    -- Storage information
    storage_location TEXT DEFAULT 'local', -- local, cloud, backup
    is_compressed BOOLEAN DEFAULT FALSE,
    compression_ratio REAL DEFAULT 1.0,
    
    -- Access control
    is_encrypted BOOLEAN DEFAULT FALSE,
    access_level TEXT DEFAULT 'private' CHECK (access_level IN ('public', 'private', 'restricted')),
    
    -- Metadata (JSON)
    metadata TEXT DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP
);

-- File info indexes for performance
CREATE INDEX idx_file_session ON file_info(session_id);
CREATE INDEX idx_file_type ON file_info(file_type);
CREATE INDEX idx_file_path ON file_info(file_path);
CREATE INDEX idx_file_created ON file_info(created_at);

-- ============================================================================
-- SYSTEM METRICS TABLE
-- ============================================================================

CREATE TABLE system_metrics (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Metric identification
    metric_name TEXT NOT NULL,
    metric_category TEXT NOT NULL,
    
    -- Metric values
    value REAL NOT NULL,
    unit TEXT,
    
    -- Context
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    component TEXT,
    
    -- Metadata (JSON)
    metadata TEXT DEFAULT '{}',
    
    -- Timestamp
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- System metrics indexes for performance
CREATE INDEX idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_metrics_category ON system_metrics(metric_category);
CREATE INDEX idx_metrics_recorded ON system_metrics(recorded_at);
CREATE INDEX idx_metrics_session ON system_metrics(session_id);

-- ============================================================================
-- DEFAULT DATA INSERTIONS
-- ============================================================================

-- Insert default configuration values
INSERT INTO configurations (category, key, value, description, data_type, is_system) VALUES
    ('system', 'version', '1.0.0', 'Application version', 'string', TRUE),
    ('system', 'database_version', '1', 'Database schema version', 'integer', TRUE),
    ('system', 'installation_date', datetime('now'), 'Installation timestamp', 'string', TRUE),
    
    ('audio', 'default_sample_rate', '44100', 'Default audio sample rate', 'integer', FALSE),
    ('audio', 'default_format', 'wav', 'Default audio format', 'string', FALSE),
    ('audio', 'quality', 'high', 'Default audio quality', 'string', FALSE),
    
    ('ai', 'whisper_model', 'base', 'Whisper model to use', 'string', FALSE),
    ('ai', 'llm_model', 'microsoft/Phi-3-mini-4k-instruct', 'LLM model for analysis', 'string', FALSE),
    ('ai', 'auto_transcribe', 'true', 'Enable automatic transcription', 'boolean', FALSE),
    ('ai', 'auto_analyze', 'true', 'Enable automatic analysis', 'boolean', FALSE),
    
    ('storage', 'max_sessions', '1000', 'Maximum number of sessions to keep', 'integer', FALSE),
    ('storage', 'retention_days', '90', 'Number of days to keep sessions', 'integer', FALSE),
    ('storage', 'auto_cleanup', 'true', 'Enable automatic cleanup', 'boolean', FALSE),
    
    ('ui', 'theme', 'dark', 'UI theme', 'string', FALSE),
    ('ui', 'language', 'en', 'UI language', 'string', FALSE),
    ('ui', 'touch_optimization', 'true', 'Enable touch optimization', 'boolean', FALSE),
    
    ('backup', 'enabled', 'true', 'Enable automatic backups', 'boolean', FALSE),
    ('backup', 'interval_hours', '6', 'Backup interval in hours', 'integer', FALSE),
    ('backup', 'max_backups', '30', 'Maximum number of backups to keep', 'integer', FALSE),
    ('backup', 'compression', 'true', 'Enable backup compression', 'boolean', FALSE);

-- Insert default user
INSERT INTO users (username, full_name, email) VALUES
    ('admin', 'System Administrator', 'admin@localhost');

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Session summary view
CREATE VIEW session_summary AS
SELECT 
    s.id,
    s.uuid,
    s.title,
    s.status,
    s.start_time,
    s.duration_seconds,
    s.participant_count,
    s.transcription_completed,
    s.analysis_completed,
    COUNT(t.id) as transcript_count,
    COUNT(a.id) as analysis_count,
    COUNT(f.id) as file_count,
    s.created_at
FROM sessions s
LEFT JOIN transcript_entries t ON s.id = t.session_id
LEFT JOIN analysis_results a ON s.id = a.session_id
LEFT JOIN file_info f ON s.id = f.session_id
GROUP BY s.id;

-- Participant statistics view
CREATE VIEW participant_statistics AS
SELECT 
    p.session_id,
    s.title as session_title,
    p.speaker_id,
    p.name,
    p.total_speaking_time_seconds,
    p.speaking_percentage,
    p.interruption_count,
    p.questions_asked,
    p.engagement_score,
    COUNT(t.id) as transcript_entries
FROM participants p
JOIN sessions s ON p.session_id = s.id
LEFT JOIN transcript_entries t ON p.session_id = t.session_id AND p.speaker_id = t.speaker_id
GROUP BY p.id;

-- Storage usage view
CREATE VIEW storage_usage AS
SELECT 
    'Sessions' as category,
    COUNT(*) as count,
    SUM(audio_size_bytes) as total_bytes,
    AVG(audio_size_bytes) as avg_bytes
FROM sessions
WHERE audio_size_bytes > 0
UNION ALL
SELECT 
    'Files' as category,
    COUNT(*) as count,
    SUM(size_bytes) as total_bytes,
    AVG(size_bytes) as avg_bytes
FROM file_info;

-- ============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- ============================================================================

-- Update session statistics when transcript entries change
CREATE TRIGGER update_session_stats_on_transcript
    AFTER INSERT ON transcript_entries
    FOR EACH ROW
BEGIN
    UPDATE sessions 
    SET 
        participant_count = (
            SELECT COUNT(DISTINCT speaker_id) 
            FROM transcript_entries 
            WHERE session_id = NEW.session_id AND speaker_id IS NOT NULL
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.session_id;
END;

-- Ensure session duration is calculated when end_time is set
CREATE TRIGGER calculate_session_duration
    AFTER UPDATE OF end_time ON sessions
    FOR EACH ROW
    WHEN NEW.end_time IS NOT NULL AND OLD.end_time IS NULL
BEGIN
    UPDATE sessions 
    SET duration_seconds = (
        CAST((julianday(NEW.end_time) - julianday(NEW.start_time)) * 86400 AS INTEGER)
    )
    WHERE id = NEW.id;
END;

-- ============================================================================
-- INDEXES FOR FULL-TEXT SEARCH (if supported)
-- ============================================================================

-- Full-text search on transcript text (SQLite FTS5 if available)
-- CREATE VIRTUAL TABLE transcript_fts USING fts5(
--     text,
--     speaker_name,
--     content='transcript_entries',
--     content_rowid='id'
-- );

-- ============================================================================
-- PERFORMANCE ANALYSIS QUERIES
-- ============================================================================

-- These are example queries for performance analysis
-- Uncomment and run as needed for optimization

-- EXPLAIN QUERY PLAN SELECT * FROM sessions WHERE status = 'completed';
-- EXPLAIN QUERY PLAN SELECT * FROM transcript_entries WHERE session_id = 1;
-- EXPLAIN QUERY PLAN SELECT * FROM sessions ORDER BY created_at DESC LIMIT 10;

-- ============================================================================
-- MAINTENANCE COMMANDS
-- ============================================================================

-- Vacuum command to reclaim space (run periodically)
-- VACUUM;

-- Analyze command to update statistics (run periodically)
-- ANALYZE;

-- Check database integrity
-- PRAGMA integrity_check;

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================

-- This table tracks schema migrations
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Final pragma settings
PRAGMA optimize;

-- Success message
SELECT 'Database schema created successfully' as result;