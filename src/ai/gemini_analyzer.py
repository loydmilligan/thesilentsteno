#!/usr/bin/env python3

"""
Gemini AI Analyzer for The Silent Steno

Enhanced AI analysis using Google's Gemini API to provide superior
meeting transcript analysis, summaries, and insights beyond simple
local analysis methods.
"""

import os
import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class GeminiAnalysisResult:
    """Structured result from Gemini analysis"""
    summary: str
    key_phrases: List[str]
    action_items: List[str]
    questions: List[str]
    topics: List[str]
    sentiment: str
    key_decisions: List[str]
    next_steps: List[str]
    participants_mentioned: List[str]
    meeting_type: str
    confidence_score: float

class GeminiAnalyzer:
    """
    Enhanced AI analysis using Google Gemini API
    
    Provides superior meeting transcript analysis including context-aware
    summaries, intelligent action item extraction, sentiment analysis,
    and meeting insights that go beyond simple keyword matching.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini analyzer
        
        Args:
            api_key: Google API key for Gemini (or via GEMINI_API_KEY env var)
            model: Gemini model to use (gemini-1.5-flash, gemini-1.5-pro)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model
        self.client = None
        self.available = False
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Request tracking
        self.requests_made = 0
        self.total_tokens_used = 0
        
        # Initialize client
        self._initialize_client()
        
        logger.info(f"Gemini analyzer initialized (model: {self.model}, available: {self.available})")
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            if not self.api_key:
                logger.warning("No Gemini API key provided - enhanced analysis unavailable")
                return
            
            import google.generativeai as genai
            
            # Configure the API
            genai.configure(api_key=self.api_key)
            
            # Create the model instance
            self.client = genai.GenerativeModel(self.model)
            self.available = True
            
            logger.info("Gemini client initialized successfully")
            
        except ImportError:
            logger.error("google-generativeai package not installed - run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.available = False
    
    def is_available(self) -> bool:
        """Check if Gemini analysis is available"""
        return self.available and self.api_key is not None
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _create_analysis_prompt(self, transcript: str) -> str:
        """Create the analysis prompt for Gemini"""
        return f"""
Analyze this meeting transcript and provide a comprehensive analysis in JSON format.

TRANSCRIPT:
{transcript}

Please provide your analysis in the following JSON structure:
{{
    "summary": "A concise 2-3 sentence summary of the meeting's main points and outcomes",
    "key_phrases": ["list", "of", "important", "phrases", "and", "terminology"],
    "action_items": ["Specific actionable tasks that were assigned or discussed"],
    "questions": ["Questions that were asked or need to be answered"],
    "topics": ["main", "topics", "discussed"],
    "sentiment": "positive/negative/neutral - overall tone of the meeting",
    "key_decisions": ["Important decisions that were made"],
    "next_steps": ["What should happen next based on this meeting"],
    "participants_mentioned": ["Names or roles of people mentioned"],
    "meeting_type": "standup/planning/review/brainstorm/formal/informal",
    "confidence_score": 0.85
}}

Guidelines for analysis:
- Focus on actionable content and concrete outcomes
- Identify genuine action items vs casual mentions
- Distinguish between decisions made vs topics discussed
- Extract participant names/roles when clearly mentioned
- Assess meeting type based on structure and content
- Provide confidence score (0.0-1.0) for analysis quality
- If transcript is unclear/short, indicate lower confidence

Respond with ONLY the JSON object, no additional text.
"""
    
    def analyze_transcript(self, transcript: str) -> Optional[GeminiAnalysisResult]:
        """
        Analyze transcript using Gemini API
        
        Args:
            transcript: The transcript text to analyze
            
        Returns:
            GeminiAnalysisResult object with analysis, or None if failed
        """
        if not self.is_available():
            logger.warning("Gemini analyzer not available")
            return None
        
        if not transcript or len(transcript.strip()) < 10:
            logger.warning("Transcript too short for Gemini analysis")
            return None
        
        try:
            # Rate limiting
            self._rate_limit()
            
            logger.info(f"Starting Gemini analysis of {len(transcript)} character transcript")
            
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(transcript)
            
            # Make the API request
            start_time = time.time()
            response = self.client.generate_content(prompt)
            analysis_time = time.time() - start_time
            
            # Track usage
            self.requests_made += 1
            
            if response.text:
                # Parse the JSON response
                try:
                    analysis_data = json.loads(response.text.strip())
                    
                    # Create structured result
                    result = GeminiAnalysisResult(
                        summary=analysis_data.get('summary', ''),
                        key_phrases=analysis_data.get('key_phrases', []),
                        action_items=analysis_data.get('action_items', []),
                        questions=analysis_data.get('questions', []),
                        topics=analysis_data.get('topics', []),
                        sentiment=analysis_data.get('sentiment', 'neutral'),
                        key_decisions=analysis_data.get('key_decisions', []),
                        next_steps=analysis_data.get('next_steps', []),
                        participants_mentioned=analysis_data.get('participants_mentioned', []),
                        meeting_type=analysis_data.get('meeting_type', 'unknown'),
                        confidence_score=analysis_data.get('confidence_score', 0.5)
                    )
                    
                    logger.info(f"Gemini analysis completed in {analysis_time:.2f}s")
                    logger.info(f"Analysis summary: {result.summary[:100]}...")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini JSON response: {e}")
                    logger.debug(f"Raw response: {response.text}")
                    return None
            else:
                logger.error("Empty response from Gemini API")
                return None
                
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None
    
    def enhance_local_analysis(self, transcript: str, local_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance local analysis with Gemini insights
        
        Args:
            transcript: Original transcript text
            local_analysis: Results from local analysis
            
        Returns:
            Enhanced analysis combining local and Gemini results
        """
        if not self.is_available():
            logger.info("Gemini not available, returning local analysis only")
            return local_analysis
        
        try:
            # Get Gemini analysis
            gemini_result = self.analyze_transcript(transcript)
            
            if not gemini_result:
                logger.warning("Gemini analysis failed, using local analysis only")
                return local_analysis
            
            # Combine local and Gemini results
            enhanced_analysis = local_analysis.copy()
            
            # Use Gemini's superior results where available
            enhanced_analysis.update({
                'summary': gemini_result.summary if gemini_result.summary else local_analysis.get('summary', ''),
                'action_items': gemini_result.action_items if gemini_result.action_items else local_analysis.get('action_items', []),
                'questions': gemini_result.questions if gemini_result.questions else local_analysis.get('questions', []),
                'topics': gemini_result.topics if gemini_result.topics else local_analysis.get('topics', []),
                'sentiment': gemini_result.sentiment if gemini_result.sentiment != 'neutral' else local_analysis.get('sentiment', 'neutral'),
                
                # Add Gemini-specific enhancements
                'key_decisions': gemini_result.key_decisions,
                'next_steps': gemini_result.next_steps,
                'participants_mentioned': gemini_result.participants_mentioned,
                'meeting_type': gemini_result.meeting_type,
                'analysis_confidence': gemini_result.confidence_score,
                'enhanced_by_gemini': True,
                
                # Keep local analysis metadata
                'key_phrases': local_analysis.get('key_phrases', []) + gemini_result.key_phrases,
                'word_count': local_analysis.get('word_count', 0),
                'duration_estimate': local_analysis.get('duration_estimate', 0)
            })
            
            logger.info("Successfully enhanced local analysis with Gemini insights")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error enhancing analysis with Gemini: {e}")
            return local_analysis
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring"""
        return {
            'requests_made': self.requests_made,
            'total_tokens_used': self.total_tokens_used,
            'available': self.available,
            'model': self.model,
            'last_request': self.last_request_time
        }

# Singleton instance for easy access
_gemini_analyzer = None

def get_gemini_analyzer(api_key: Optional[str] = None) -> GeminiAnalyzer:
    """Get the global Gemini analyzer instance"""
    global _gemini_analyzer
    if _gemini_analyzer is None:
        _gemini_analyzer = GeminiAnalyzer(api_key)
    return _gemini_analyzer

def analyze_with_gemini(transcript: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function for Gemini analysis
    
    Args:
        transcript: Text to analyze
        api_key: Optional API key
        
    Returns:
        Analysis results as dictionary or None if failed
    """
    analyzer = get_gemini_analyzer(api_key)
    result = analyzer.analyze_transcript(transcript)
    
    if result:
        return {
            'summary': result.summary,
            'action_items': result.action_items,
            'questions': result.questions,
            'topics': result.topics,
            'sentiment': result.sentiment,
            'key_decisions': result.key_decisions,
            'next_steps': result.next_steps,
            'participants_mentioned': result.participants_mentioned,
            'meeting_type': result.meeting_type,
            'confidence_score': result.confidence_score,
            'enhanced_by_gemini': True
        }
    
    return None

if __name__ == "__main__":
    # Test the Gemini analyzer
    print("Gemini Analyzer Test")
    print("=" * 50)
    
    # Test with sample transcript
    sample_transcript = """
    Good morning everyone. Let's start today's standup. 
    John, can you give us an update on the user authentication feature?
    
    John: Sure, I completed the login functionality yesterday. The OAuth integration is working well.
    I'm planning to start on the password reset feature today. Should be done by Thursday.
    
    Sarah: Great! I reviewed the database schema changes you made. They look good. 
    I'll merge your pull request after this meeting.
    
    John: Thanks Sarah. One question - should we implement two-factor authentication as well?
    
    Manager: That's a good point. Let's add that to our backlog for next sprint. 
    Sarah, can you create a ticket for that?
    
    Sarah: Will do. I'll also schedule a security review meeting with the team.
    
    Manager: Perfect. Any blockers for anyone?
    
    John: No blockers for me.
    
    Sarah: I'm waiting for the design team to finalize the new UI mockups, 
    but I have other tasks to work on in the meantime.
    
    Manager: Okay, I'll follow up with the design team today. 
    Let's wrap up. Thanks everyone!
    """
    
    # Create analyzer (will need API key to actually test)
    analyzer = GeminiAnalyzer()
    print(f"Analyzer available: {analyzer.is_available()}")
    
    if analyzer.is_available():
        print("\nAnalyzing sample transcript...")
        result = analyzer.analyze_transcript(sample_transcript)
        
        if result:
            print(f"Summary: {result.summary}")
            print(f"Action Items: {result.action_items}")
            print(f"Meeting Type: {result.meeting_type}")
            print(f"Confidence: {result.confidence_score}")
        else:
            print("Analysis failed")
    else:
        print("Set GEMINI_API_KEY environment variable to test with real API")
    
    print("\nTest complete")