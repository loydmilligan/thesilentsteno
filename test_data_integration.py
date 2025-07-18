#!/usr/bin/env python3

"""
Test script for data integration functionality
"""

import os
import sys
sys.path.append('/home/mmariani/projects/thesilentsteno/src')

from data.integration_adapter import DataIntegrationAdapter
from recording.simple_audio_recorder import SimpleAudioRecorder
from datetime import datetime
import json

def test_data_integration():
    print("Testing Data Integration")
    print("=" * 50)
    
    # Test 1: Create data adapter
    print("\n1. Testing DataIntegrationAdapter...")
    adapter = DataIntegrationAdapter("demo_sessions/sessions.json", use_database=False)
    
    # Get current sessions
    sessions = adapter.get_all_sessions()
    print(f"   Found {len(sessions)} existing sessions")
    
    # Test 2: Create new session
    print("\n2. Testing session creation...")
    test_session = {
        'session_id': 'test-integration-' + str(datetime.now().timestamp()),
        'title': 'Data Integration Test Session',
        'timestamp': datetime.now().isoformat(),
        'duration': 5.0,
        'transcript': 'This is a test transcript for data integration'
    }
    
    session_id = adapter.create_session(test_session)
    if session_id:
        print(f"   ✓ Created session: {session_id}")
    else:
        print("   ✗ Failed to create session")
        return False
    
    # Test 3: Retrieve session
    print("\n3. Testing session retrieval...")
    retrieved_session = adapter.get_session(session_id)
    if retrieved_session:
        print(f"   ✓ Retrieved session: {retrieved_session['title']}")
        print(f"   ✓ Transcript: {retrieved_session['transcript']}")
    else:
        print("   ✗ Failed to retrieve session")
        return False
    
    # Test 4: Update session
    print("\n4. Testing session update...")
    update_success = adapter.update_session(session_id, {'transcript': 'Updated transcript for integration test'})
    if update_success:
        print("   ✓ Session updated successfully")
        
        # Verify update
        updated_session = adapter.get_session(session_id)
        if updated_session and 'Updated transcript' in updated_session['transcript']:
            print("   ✓ Update verified")
        else:
            print("   ✗ Update verification failed")
            return False
    else:
        print("   ✗ Failed to update session")
        return False
    
    # Test 5: Storage stats
    print("\n5. Testing storage statistics...")
    stats = adapter.get_storage_stats()
    print(f"   Storage type: {stats['storage_type']}")
    print(f"   Session count: {stats['session_count']}")
    print(f"   Transcript count: {stats['transcript_count']}")
    print(f"   Database available: {stats['database_available']}")
    
    # Test 6: SimpleAudioRecorder integration
    print("\n6. Testing SimpleAudioRecorder integration...")
    recorder = SimpleAudioRecorder("demo_sessions")
    
    # Check if data adapter is initialized
    if hasattr(recorder, 'data_adapter') and recorder.data_adapter:
        print("   ✓ Data adapter initialized in recorder")
        
        # Test loading sessions
        recorder_sessions = recorder.get_all_sessions()
        print(f"   ✓ Loaded {len(recorder_sessions)} sessions via recorder")
        
        # Verify our test session is in the list
        test_session_found = False
        for session in recorder_sessions:
            if session.get('session_id') == session_id:
                test_session_found = True
                break
        
        if test_session_found:
            print("   ✓ Test session found in recorder sessions")
        else:
            print("   ✗ Test session not found in recorder sessions")
            return False
    else:
        print("   ✗ Data adapter not initialized in recorder")
        return False
    
    print("\n✓ All data integration tests passed!")
    return True

def test_json_file_format():
    print("\nTesting JSON file format consistency...")
    
    # Check existing demo sessions
    sessions_file = "demo_sessions/sessions.json"
    if os.path.exists(sessions_file):
        with open(sessions_file, 'r') as f:
            sessions = json.load(f)
        
        print(f"Found {len(sessions)} existing sessions")
        
        # Check session structure
        if sessions:
            sample_session = sessions[0]
            print(f"Sample session keys: {list(sample_session.keys())}")
            
            # Check for expected fields
            required_fields = ['timestamp', 'wav_file', 'duration']
            for field in required_fields:
                if field in sample_session:
                    print(f"   ✓ {field}: {sample_session[field]}")
                else:
                    print(f"   ✗ Missing field: {field}")
            
            # Check for newer fields
            newer_fields = ['session_id', 'title', 'sample_rate', 'channels']
            for field in newer_fields:
                if field in sample_session:
                    print(f"   ✓ {field}: {sample_session[field]}")
                else:
                    print(f"   - {field}: not present (ok for older sessions)")
    else:
        print("No existing sessions file found")

if __name__ == "__main__":
    print("Data Integration Test Suite")
    print("=" * 50)
    
    # Test JSON file format first
    test_json_file_format()
    
    # Test data integration
    success = test_data_integration()
    
    if success:
        print("\n🎉 All tests passed! Data integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.")