#!/usr/bin/env python3
"""
Generate synthetic test audio for The Silent Steno
Creates realistic meeting, work call, and personal call audio samples
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Try to import required libraries
try:
    from gtts import gTTS
    import pydub
    from pydub import AudioSegment
    from pydub.generators import WhiteNoise
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError as e:
    print("Missing dependencies. Please install:")
    print("pip install gtts pydub nltk")
    sys.exit(1)

class TestAudioGenerator:
    """Generate synthetic test audio for various scenarios"""
    
    def __init__(self, output_dir: str = "test_audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Voice configurations for different speakers
        self.voices = {
            "sarah": {"lang": "en", "tld": "com", "slow": False},     # American female
            "john": {"lang": "en", "tld": "co.uk", "slow": False},    # British male
            "mike": {"lang": "en", "tld": "com.au", "slow": False},   # Australian male
            "emily": {"lang": "en", "tld": "ca", "slow": False},      # Canadian female
            "raj": {"lang": "en", "tld": "co.in", "slow": False},     # Indian male
        }
        
        # Background noise profiles
        self.noise_profiles = {
            "office": {"volume": -35, "variations": True},
            "home": {"volume": -40, "variations": False},
            "coffee_shop": {"volume": -30, "variations": True},
            "car": {"volume": -32, "variations": True},
            "quiet": {"volume": -45, "variations": False},
        }
    
    def generate_meeting_script(self) -> List[Dict[str, str]]:
        """Generate a realistic meeting script"""
        return [
            {"speaker": "sarah", "text": "Good morning everyone. Let's get started with our weekly team sync."},
            {"speaker": "sarah", "text": "I'd like to begin by reviewing last week's action items."},
            {"speaker": "john", "text": "Thanks Sarah. I completed the API integration for the payment system."},
            {"speaker": "john", "text": "We're seeing about 200 millisecond response times, which is within our target."},
            {"speaker": "mike", "text": "That's great John. I've been working on the frontend components."},
            {"speaker": "mike", "text": "The user dashboard is almost complete. Should be ready for review by Thursday."},
            {"speaker": "sarah", "text": "Excellent progress. Emily, how's the testing framework coming along?"},
            {"speaker": "emily", "text": "We've got about 80 percent coverage now. I'm focusing on edge cases this week."},
            {"speaker": "emily", "text": "I did find one issue with the authentication flow that we need to discuss."},
            {"speaker": "sarah", "text": "Let's make that a priority. Can you work with John on that?"},
            {"speaker": "john", "text": "Sure, I can look at that this afternoon."},
            {"speaker": "raj", "text": "Sorry I'm late everyone. Just wanted to mention the deployment pipeline is fully automated now."},
            {"speaker": "sarah", "text": "No problem Raj. That's great news about the pipeline."},
            {"speaker": "sarah", "text": "Okay, let's talk about next week's priorities."},
            {"speaker": "mike", "text": "I think we should focus on performance optimization."},
            {"speaker": "john", "text": "Agreed. The database queries could use some work."},
            {"speaker": "emily", "text": "I'll add performance tests to our suite."},
            {"speaker": "sarah", "text": "Perfect. Let's set up a follow-up on Wednesday to check progress."},
            {"speaker": "sarah", "text": "Any blockers or concerns before we wrap up?"},
            {"speaker": "raj", "text": "Just need the AWS credentials updated for the staging environment."},
            {"speaker": "sarah", "text": "I'll handle that right after this call. Anything else?"},
            {"speaker": "sarah", "text": "Alright, great meeting everyone. Let's stay focused on these priorities."},
            {"speaker": "all", "text": "[mixed voices saying goodbye]"},
        ]
    
    def generate_work_call_script(self) -> List[Dict[str, str]]:
        """Generate a work phone call script"""
        return [
            {"speaker": "john", "text": "Hi Sarah, do you have a minute to discuss the client proposal?"},
            {"speaker": "sarah", "text": "Sure John, I was just reviewing it actually. What's on your mind?"},
            {"speaker": "john", "text": "I'm concerned about the timeline. Six weeks seems aggressive for this scope."},
            {"speaker": "sarah", "text": "I see your point. What specific parts are you worried about?"},
            {"speaker": "john", "text": "The integration with their legacy system could be tricky. We don't know what we'll find."},
            {"speaker": "sarah", "text": "That's true. Should we add a discovery phase first?"},
            {"speaker": "john", "text": "That would help. Maybe two weeks for discovery and assessment?"},
            {"speaker": "sarah", "text": "Let me check the budget... Yes, we have room for that."},
            {"speaker": "john", "text": "Great. Also, we'll need their IT team available during that phase."},
            {"speaker": "sarah", "text": "I'll add that to the requirements. What about resources on our end?"},
            {"speaker": "john", "text": "I can dedicate Mike and Emily to this project."},
            {"speaker": "sarah", "text": "Perfect. They did great work on the previous integration."},
            {"speaker": "john", "text": "Exactly. Oh, one more thing - the security requirements look extensive."},
            {"speaker": "sarah", "text": "Yes, they're in financial services so compliance is critical."},
            {"speaker": "john", "text": "We might need to bring in our security consultant."},
            {"speaker": "sarah", "text": "Good thinking. I'll reach out to David today."},
            {"speaker": "john", "text": "Sounds good. When do you want to send the revised proposal?"},
            {"speaker": "sarah", "text": "Let's aim for end of day tomorrow. Can you update the technical section?"},
            {"speaker": "john", "text": "Absolutely. I'll have it to you by lunch tomorrow."},
            {"speaker": "sarah", "text": "Perfect. Thanks for flagging these issues, John."},
            {"speaker": "john", "text": "No problem. Talk to you tomorrow."},
            {"speaker": "sarah", "text": "Bye!"},
        ]
    
    def generate_personal_call_script(self) -> List[Dict[str, str]]:
        """Generate a personal phone call script"""
        return [
            {"speaker": "emily", "text": "Hey Mike! How's it going?"},
            {"speaker": "mike", "text": "Emily! Good to hear from you. I'm doing well, just got back from vacation."},
            {"speaker": "emily", "text": "Oh nice! Where did you go?"},
            {"speaker": "mike", "text": "We went to Hawaii. First time visiting and it was amazing."},
            {"speaker": "emily", "text": "That sounds wonderful! Did you do any surfing?"},
            {"speaker": "mike", "text": "I tried! Let's just say I spent more time underwater than on the board."},
            {"speaker": "emily", "text": "[laughing] At least you gave it a shot! How's Sarah doing?"},
            {"speaker": "mike", "text": "She's great. Actually, she got promoted last month to senior manager."},
            {"speaker": "emily", "text": "That's fantastic! She definitely deserved it."},
            {"speaker": "mike", "text": "Yeah, we're pretty excited. How about you? How's the new job?"},
            {"speaker": "emily", "text": "It's been a learning curve but I'm really enjoying it."},
            {"speaker": "emily", "text": "The team is super supportive and the projects are interesting."},
            {"speaker": "mike", "text": "That's what matters. Are you still playing tennis?"},
            {"speaker": "emily", "text": "Every weekend! Actually just joined a league."},
            {"speaker": "mike", "text": "Nice! We should play doubles sometime when I visit."},
            {"speaker": "emily", "text": "Definitely! When are you planning to come to town?"},
            {"speaker": "mike", "text": "Probably next month for Tom's birthday. You're going, right?"},
            {"speaker": "emily", "text": "Wouldn't miss it! It'll be great to see everyone."},
            {"speaker": "mike", "text": "For sure. Hey, I should probably get going. Meeting in five minutes."},
            {"speaker": "emily", "text": "No worries! Great catching up. Give my best to Sarah!"},
            {"speaker": "mike", "text": "Will do! Talk soon!"},
            {"speaker": "emily", "text": "Bye!"},
        ]
    
    def create_audio_segment(self, text: str, speaker: str, 
                           add_pause: bool = True) -> AudioSegment:
        """Create an audio segment for a single utterance"""
        voice_config = self.voices.get(speaker, self.voices["sarah"])
        
        # Generate speech
        tts = gTTS(text=text, lang=voice_config["lang"], 
                   tld=voice_config["tld"], slow=voice_config["slow"])
        
        # Save to temporary file
        temp_file = f"temp_{speaker}_{random.randint(1000, 9999)}.mp3"
        tts.save(temp_file)
        
        # Load and process audio
        audio = AudioSegment.from_mp3(temp_file)
        
        # Adjust pitch slightly for different speakers
        if speaker == "john" or speaker == "mike" or speaker == "raj":
            # Lower pitch for male voices
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * 0.9)
            }).set_frame_rate(audio.frame_rate)
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Add natural pause after speech
        if add_pause:
            pause_duration = random.randint(300, 800)  # 0.3 to 0.8 seconds
            pause = AudioSegment.silent(duration=pause_duration)
            audio = audio + pause
        
        return audio
    
    def add_background_noise(self, audio: AudioSegment, 
                           noise_type: str = "office") -> AudioSegment:
        """Add realistic background noise to audio"""
        noise_config = self.noise_profiles.get(noise_type, self.noise_profiles["quiet"])
        
        # Generate noise
        noise = WhiteNoise().to_audio_segment(duration=len(audio))
        noise = noise + noise_config["volume"]  # Reduce volume
        
        # Add variations if specified
        if noise_config["variations"]:
            # Add occasional louder sounds (keyboard, cough, etc.)
            for _ in range(random.randint(2, 5)):
                position = random.randint(0, len(audio) - 1000)
                burst = WhiteNoise().to_audio_segment(duration=random.randint(100, 300))
                burst = burst + (noise_config["volume"] + 10)
                noise = noise.overlay(burst, position=position)
        
        # Mix with original audio
        return audio.overlay(noise)
    
    def generate_audio_file(self, script: List[Dict[str, str]], 
                          filename: str, noise_type: str = "office",
                          sample_rate: int = 16000) -> str:
        """Generate complete audio file from script"""
        print(f"Generating {filename}...")
        
        # Create all audio segments
        segments = []
        for i, line in enumerate(script):
            print(f"  Processing line {i+1}/{len(script)}: {line['speaker']}")
            
            if line["speaker"] == "all":
                # Handle multiple speakers talking at once
                overlapping = AudioSegment.silent(duration=2000)
                for speaker in ["sarah", "john", "mike", "emily"]:
                    goodbye = self.create_audio_segment("Bye everyone!", speaker, add_pause=False)
                    position = random.randint(0, 500)
                    overlapping = overlapping.overlay(goodbye, position=position)
                segments.append(overlapping)
            else:
                segment = self.create_audio_segment(line["text"], line["speaker"])
                segments.append(segment)
        
        # Combine all segments
        full_audio = AudioSegment.empty()
        for segment in segments:
            full_audio += segment
        
        # Add background noise
        full_audio = self.add_background_noise(full_audio, noise_type)
        
        # Convert to target sample rate and save
        full_audio = full_audio.set_frame_rate(sample_rate)
        output_path = os.path.join(self.output_dir, filename)
        
        # Export as WAV for better compatibility
        full_audio.export(output_path, format="wav")
        
        # Generate metadata
        metadata = {
            "filename": filename,
            "duration_seconds": len(full_audio) / 1000,
            "sample_rate": sample_rate,
            "noise_type": noise_type,
            "speakers": list(set(line["speaker"] for line in script if line["speaker"] != "all")),
            "utterances": len(script),
            "generated_at": datetime.now().isoformat()
        }
        
        metadata_path = output_path.replace('.wav', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Generated: {output_path}")
        print(f"  ✓ Duration: {metadata['duration_seconds']:.1f} seconds")
        print(f"  ✓ Metadata: {metadata_path}")
        
        return output_path
    
    def generate_test_suite(self):
        """Generate complete test audio suite"""
        print("Generating Silent Steno Test Audio Suite")
        print("=" * 50)
        
        # Generate meeting audio (longest, most complex)
        meeting_script = self.generate_meeting_script()
        self.generate_audio_file(meeting_script, "test_meeting_5min.wav", 
                               noise_type="office")
        
        # Generate work call (medium length, two speakers)
        work_script = self.generate_work_call_script()
        self.generate_audio_file(work_script, "test_work_call_3min.wav", 
                               noise_type="quiet")
        
        # Generate personal call (casual, background noise)
        personal_script = self.generate_personal_call_script()
        self.generate_audio_file(personal_script, "test_personal_call_2min.wav", 
                               noise_type="home")
        
        # Generate challenging audio scenarios
        print("\nGenerating challenging test cases...")
        
        # 1. Noisy environment
        noisy_script = self.generate_meeting_script()[:10]  # Shorter version
        self.generate_audio_file(noisy_script, "test_noisy_coffee_shop.wav", 
                               noise_type="coffee_shop")
        
        # 2. Fast speakers (for stress testing)
        fast_script = [
            {"speaker": "sarah", "text": "We need to move quickly on this project timeline is critical"},
            {"speaker": "john", "text": "Absolutely I'll coordinate with the team immediately"},
            {"speaker": "sarah", "text": "Great make sure to loop in Emily and Mike they have the context"},
            {"speaker": "john", "text": "Will do I'll set up a meeting for this afternoon"},
        ]
        self.generate_audio_file(fast_script, "test_fast_speech.wav", 
                               noise_type="quiet")
        
        # Create test manifest
        manifest = {
            "test_suite": "Silent Steno Audio Test Suite",
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "files": [
                {
                    "name": "test_meeting_5min.wav",
                    "description": "5-person team meeting with multiple speakers",
                    "use_case": "Testing speaker diarization and meeting analysis",
                    "expected_features": ["action_items", "multiple_speakers", "technical_discussion"]
                },
                {
                    "name": "test_work_call_3min.wav",
                    "description": "Two-person work phone call about project planning",
                    "use_case": "Testing phone call transcription and business context",
                    "expected_features": ["project_discussion", "timeline_planning", "decision_making"]
                },
                {
                    "name": "test_personal_call_2min.wav",
                    "description": "Casual personal phone call between friends",
                    "use_case": "Testing informal speech and personal context detection",
                    "expected_features": ["casual_conversation", "personal_topics", "emotional_content"]
                },
                {
                    "name": "test_noisy_coffee_shop.wav",
                    "description": "Meeting audio with significant background noise",
                    "use_case": "Testing noise robustness and audio quality handling",
                    "expected_features": ["background_noise", "multiple_speakers", "challenging_audio"]
                },
                {
                    "name": "test_fast_speech.wav",
                    "description": "Fast-paced conversation without pauses",
                    "use_case": "Testing real-time transcription performance",
                    "expected_features": ["rapid_speech", "no_pauses", "stress_test"]
                }
            ]
        }
        
        manifest_path = os.path.join(self.output_dir, "test_audio_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Test suite generated in: {self.output_dir}/")
        print(f"✓ Manifest saved to: {manifest_path}")
        
def main():
    parser = argparse.ArgumentParser(description="Generate test audio for Silent Steno")
    parser.add_argument("--output-dir", default="test_audio", 
                       help="Output directory for test files")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Sample rate for audio files (default: 16000)")
    
    args = parser.parse_args()
    
    generator = TestAudioGenerator(output_dir=args.output_dir)
    generator.generate_test_suite()

if __name__ == "__main__":
    main()