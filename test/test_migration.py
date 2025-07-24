#!/usr/bin/env python3

"""
Test script for database migration functionality
"""

import os
import sys
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

from data.integration_adapter import DataIntegrationAdapter

def test_migration():
    print("Testing Database Migration")
    print("=" * 50)
    
    # Test 1: Create adapter in JSON mode to see existing sessions
    print("\n1. Testing JSON sessions before migration...")
    json_adapter = DataIntegrationAdapter("demo_sessions/sessions.json", use_database=False)
    json_sessions = json_adapter.get_all_sessions()
    print(f"   Found {len(json_sessions)} sessions in JSON")
    
    # Test 2: Create adapter in database mode to see current database state
    print("\n2. Testing database sessions before migration...")
    db_adapter = DataIntegrationAdapter("demo_sessions/sessions.json", use_database=True)
    db_sessions = db_adapter.get_all_sessions()
    print(f"   Found {len(db_sessions)} sessions in database")
    
    # Test 3: Migrate from JSON to database
    print("\n3. Testing migration from JSON to database...")
    success = db_adapter.migrate_to_database()
    if success:
        print("   ✓ Migration completed successfully")
        
        # Test 4: Verify migration
        print("\n4. Verifying migration...")
        db_sessions_after = db_adapter.get_all_sessions()
        print(f"   Found {len(db_sessions_after)} sessions in database after migration")
        
        # Show sample migrated session
        if db_sessions_after:
            sample = db_sessions_after[0]
            print(f"   Sample session: {sample['title']}")
            print(f"   Transcript: {sample.get('transcript', 'No transcript')}")
            print(f"   Duration: {sample.get('duration', 0):.2f}s")
    else:
        print("   ✗ Migration failed")
        return False
    
    # Test 5: Test new session creation in database mode
    print("\n5. Testing new session creation in database mode...")
    new_session = {
        'session_id': 'test-db-session',
        'title': 'Database Test Session',
        'duration': 10.0,
        'transcript': 'This is a test transcript in database mode'
    }
    
    session_id = db_adapter.create_session(new_session)
    if session_id:
        print(f"   ✓ Created database session: {session_id}")
        
        # Verify it's in database
        retrieved = db_adapter.get_session(session_id)
        if retrieved:
            print(f"   ✓ Retrieved session: {retrieved['title']}")
        else:
            print("   ✗ Failed to retrieve created session")
            return False
    else:
        print("   ✗ Failed to create database session")
        return False
    
    # Test 6: Test recorder with database mode
    print("\n6. Testing recorder with database mode...")
    from recording.simple_audio_recorder import SimpleAudioRecorder
    
    recorder = SimpleAudioRecorder("demo_sessions")
    if recorder.data_adapter and recorder.data_adapter.use_database:
        print("   ✓ Recorder using database mode")
        
        recorder_sessions = recorder.get_all_sessions()
        print(f"   ✓ Recorder loaded {len(recorder_sessions)} sessions from database")
        
        if len(recorder_sessions) > 0:
            print("   ✓ Sessions successfully loaded from database")
        else:
            print("   ✗ No sessions loaded from database")
            return False
    else:
        print("   ✗ Recorder not using database mode")
        return False
    
    print("\n✓ All migration tests passed!")
    return True

if __name__ == "__main__":
    success = test_migration()
    
    if success:
        print("\n🎉 Database migration is working correctly!")
    else:
        print("\n❌ Migration tests failed.")