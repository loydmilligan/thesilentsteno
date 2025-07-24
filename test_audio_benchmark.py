#!/usr/bin/env python3
"""
Automated testing and benchmarking for The Silent Steno
Uses synthetic test audio to measure performance and accuracy
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append('/home/mmariani/projects/thesilentsteno')

from src.integration.walking_skeleton_adapter import create_walking_skeleton_adapter
from src.config.settings_manager import get_settings_manager

class TranscriptionBenchmark:
    """Benchmark transcription performance and accuracy"""
    
    def __init__(self, test_audio_dir: str = "test_audio"):
        self.test_audio_dir = test_audio_dir
        self.results = []
        self.adapter = None
        
    def setup(self):
        """Initialize the adapter and settings"""
        print("Setting up benchmark environment...")
        self.adapter = create_walking_skeleton_adapter(use_production=False)
        self.adapter.initialize()
        
    def load_test_manifest(self) -> Dict:
        """Load the test audio manifest"""
        manifest_path = os.path.join(self.test_audio_dir, "test_audio_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Test manifest not found at {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def benchmark_transcription(self, audio_file: str, 
                              expected_features: List[str]) -> Dict:
        """Benchmark a single audio file"""
        print(f"\nBenchmarking: {audio_file}")
        print("-" * 50)
        
        audio_path = os.path.join(self.test_audio_dir, audio_file)
        if not os.path.exists(audio_path):
            print(f"  ✗ Audio file not found: {audio_path}")
            return None
        
        # Load metadata if available
        metadata_path = audio_path.replace('.wav', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Start transcription
        start_time = time.time()
        print("  → Starting transcription...")
        
        try:
            # Transcribe using the adapter
            result = self.adapter.transcribe_recording(audio_path)
            transcription_time = time.time() - start_time
            
            if result:
                if isinstance(result, dict):
                    transcript = result.get('transcript', '')
                    analysis = result.get('analysis', {})
                else:
                    transcript = str(result)
                    analysis = {}
                
                # Calculate metrics
                audio_duration = metadata.get('duration_seconds', 0)
                real_time_factor = transcription_time / audio_duration if audio_duration > 0 else 0
                
                # Check for expected features
                features_found = []
                if "action_items" in expected_features and analysis.get('action_items'):
                    features_found.append("action_items")
                if "multiple_speakers" in expected_features and metadata.get('speakers', []):
                    features_found.append("multiple_speakers")
                if "technical_discussion" in expected_features and any(
                    term in transcript.lower() for term in ['api', 'integration', 'deployment', 'testing']):
                    features_found.append("technical_discussion")
                
                # Build result
                benchmark_result = {
                    "file": audio_file,
                    "success": True,
                    "transcription_time": round(transcription_time, 2),
                    "audio_duration": round(audio_duration, 2),
                    "real_time_factor": round(real_time_factor, 2),
                    "transcript_length": len(transcript),
                    "word_count": len(transcript.split()),
                    "expected_features": expected_features,
                    "features_found": features_found,
                    "feature_detection_rate": len(features_found) / len(expected_features) if expected_features else 0,
                    "analysis_available": bool(analysis),
                    "summary_length": len(analysis.get('summary', '')),
                    "action_items_count": len(analysis.get('action_items', [])),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Print results
                print(f"  ✓ Transcription completed in {transcription_time:.2f}s")
                print(f"  ✓ Real-time factor: {real_time_factor:.2f}x")
                print(f"  ✓ Words transcribed: {benchmark_result['word_count']}")
                print(f"  ✓ Features detected: {len(features_found)}/{len(expected_features)}")
                
                if real_time_factor > 1.0:
                    print(f"  ⚠ Slower than real-time by {(real_time_factor - 1) * 100:.1f}%")
                else:
                    print(f"  ✓ Faster than real-time by {(1 - real_time_factor) * 100:.1f}%")
                
                return benchmark_result
                
            else:
                print("  ✗ Transcription failed - no result returned")
                return {
                    "file": audio_file,
                    "success": False,
                    "error": "No transcription result",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"  ✗ Transcription error: {str(e)}")
            return {
                "file": audio_file,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("Silent Steno Transcription Benchmark Suite")
        print("="*60)
        
        # Load test manifest
        try:
            manifest = self.load_test_manifest()
            print(f"\nLoaded test suite: {manifest['test_suite']}")
            print(f"Version: {manifest['version']}")
            print(f"Test files: {len(manifest['files'])}")
        except FileNotFoundError as e:
            print(f"\n✗ {e}")
            print("Please run generate_test_audio.py first to create test files.")
            return
        
        # Setup environment
        self.setup()
        
        # Run benchmarks
        for test_file in manifest['files']:
            result = self.benchmark_transcription(
                test_file['name'],
                test_file.get('expected_features', [])
            )
            if result:
                result['description'] = test_file['description']
                result['use_case'] = test_file['use_case']
                self.results.append(result)
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "="*60)
        print("Benchmark Summary Report")
        print("="*60)
        
        if not self.results:
            print("No results to report.")
            return
        
        # Calculate aggregate metrics
        successful_tests = [r for r in self.results if r.get('success', False)]
        failed_tests = [r for r in self.results if not r.get('success', False)]
        
        print(f"\nTests completed: {len(self.results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if successful_tests:
            avg_rtf = sum(r.get('real_time_factor', 0) for r in successful_tests) / len(successful_tests)
            avg_time = sum(r.get('transcription_time', 0) for r in successful_tests) / len(successful_tests)
            total_words = sum(r.get('word_count', 0) for r in successful_tests)
            avg_feature_detection = sum(r.get('feature_detection_rate', 0) for r in successful_tests) / len(successful_tests)
            
            print(f"\nPerformance Metrics:")
            print(f"  Average real-time factor: {avg_rtf:.2f}x")
            print(f"  Average transcription time: {avg_time:.2f}s")
            print(f"  Total words transcribed: {total_words}")
            print(f"  Average feature detection: {avg_feature_detection:.1%}")
            
            # Performance rating
            if avg_rtf < 0.5:
                rating = "Excellent (2x+ faster than real-time)"
            elif avg_rtf < 1.0:
                rating = "Good (faster than real-time)"
            elif avg_rtf < 1.5:
                rating = "Acceptable (slightly slower than real-time)"
            else:
                rating = "Needs optimization (significantly slower than real-time)"
            
            print(f"\nPerformance Rating: {rating}")
        
        # Save detailed report
        report_path = os.path.join(self.test_audio_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_data = {
            "benchmark_suite": "Silent Steno Transcription Benchmark",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "average_real_time_factor": avg_rtf if successful_tests else 0,
                "performance_rating": rating if successful_tests else "No successful tests"
            },
            "detailed_results": self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print individual test results
        print("\nIndividual Test Results:")
        print("-" * 60)
        for result in self.results:
            print(f"\n{result['file']}:")
            print(f"  Description: {result.get('description', 'N/A')}")
            if result.get('success'):
                print(f"  ✓ Success - RTF: {result.get('real_time_factor', 0):.2f}x, "
                      f"Words: {result.get('word_count', 0)}, "
                      f"Features: {len(result.get('features_found', []))}/{len(result.get('expected_features', []))}")
            else:
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Silent Steno transcription")
    parser.add_argument("--test-dir", default="test_audio", 
                       help="Directory containing test audio files")
    parser.add_argument("--single-file", help="Benchmark a single file only")
    
    args = parser.parse_args()
    
    benchmark = TranscriptionBenchmark(test_audio_dir=args.test_dir)
    
    if args.single_file:
        # Run single file benchmark
        benchmark.setup()
        result = benchmark.benchmark_transcription(args.single_file, [])
        if result:
            print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        # Run full suite
        benchmark.run_benchmark_suite()

if __name__ == "__main__":
    main()