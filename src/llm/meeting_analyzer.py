#!/usr/bin/env python3
"""
Meeting Analyzer Module

Meeting analysis engine with comprehensive summarization, insights extraction,
and participant analysis capabilities.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
from datetime import datetime

# Local imports
from .local_llm_processor import LocalLLMProcessor, LLMConfig, LLMResult
from .prompt_templates import PromptTemplateManager, TemplateType, MeetingType, OutputFormat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of meeting analysis"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"


class MeetingPhase(Enum):
    """Meeting phases for analysis"""
    OPENING = "opening"
    MAIN_DISCUSSION = "main_discussion"
    DECISION_MAKING = "decision_making"
    CLOSING = "closing"
    FULL_MEETING = "full_meeting"


@dataclass
class AnalysisConfig:
    """Configuration for meeting analysis"""
    
    # Analysis settings
    analysis_type: AnalysisType = AnalysisType.SUMMARY
    meeting_type: MeetingType = MeetingType.GENERAL
    
    # Content settings
    include_summary: bool = True
    include_action_items: bool = True
    include_topics: bool = True
    include_participants: bool = True
    include_insights: bool = False
    include_sentiment: bool = False
    
    # Output settings
    output_format: OutputFormat = OutputFormat.STRUCTURED
    max_summary_length: int = 500
    max_action_items: int = 10
    max_topics: int = 5
    
    # Processing settings
    chunk_size: int = 2000  # Characters per chunk
    overlap_size: int = 200  # Overlap between chunks
    
    # Quality settings
    min_confidence: float = 0.6
    require_validation: bool = True
    
    # Context settings
    context_information: str = ""
    custom_instructions: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "analysis_type": self.analysis_type.value,
            "meeting_type": self.meeting_type.value,
            "include_summary": self.include_summary,
            "include_action_items": self.include_action_items,
            "include_topics": self.include_topics,
            "include_participants": self.include_participants,
            "include_insights": self.include_insights,
            "include_sentiment": self.include_sentiment,
            "output_format": self.output_format.value,
            "max_summary_length": self.max_summary_length,
            "max_action_items": self.max_action_items,
            "max_topics": self.max_topics,
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "min_confidence": self.min_confidence,
            "require_validation": self.require_validation,
            "context_information": self.context_information,
            "custom_instructions": self.custom_instructions
        }


@dataclass
class AnalysisResult:
    """Result from meeting analysis"""
    
    # Analysis metadata
    analysis_id: str
    analysis_type: AnalysisType
    meeting_type: MeetingType
    
    # Timing information
    start_time: datetime
    end_time: datetime
    processing_time: float
    
    # Analysis components
    summary: str = ""
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[Dict[str, Any]] = field(default_factory=list)
    insights: Dict[str, Any] = field(default_factory=dict)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    confidence: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    
    # Input metadata
    input_text: str = ""
    input_length: int = 0
    chunks_processed: int = 0
    
    # Processing metadata
    llm_results: List[LLMResult] = field(default_factory=list)
    
    # Status
    success: bool = True
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type.value,
            "meeting_type": self.meeting_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "processing_time": self.processing_time,
            "summary": self.summary,
            "action_items": self.action_items,
            "topics": self.topics,
            "participants": self.participants,
            "insights": self.insights,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "input_length": self.input_length,
            "chunks_processed": self.chunks_processed,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class MeetingAnalyzer:
    """Main meeting analysis engine"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.llm_processor = None
        self.template_manager = None
        self.is_initialized = False
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "error_count": 0,
            "success_rate": 0.0
        }
        
        # Cache for repeated analyses
        self.analysis_cache = {}
        
        logger.info("MeetingAnalyzer initialized")
        
    def initialize(self, llm_processor: Optional[LocalLLMProcessor] = None) -> bool:
        """Initialize the meeting analyzer"""
        try:
            logger.info("Initializing meeting analyzer...")
            
            # Initialize LLM processor
            if llm_processor:
                self.llm_processor = llm_processor
            else:
                from .local_llm_processor import create_meeting_processor
                self.llm_processor = create_meeting_processor()
                
            if not self.llm_processor.is_initialized:
                if not self.llm_processor.initialize():
                    logger.error("Failed to initialize LLM processor")
                    return False
                    
            # Initialize template manager
            from .prompt_templates import create_analysis_templates
            self.template_manager = create_analysis_templates()
            
            # Update template manager config
            self.template_manager.config.meeting_type = self.config.meeting_type
            self.template_manager.config.output_format = self.config.output_format
            self.template_manager.config.max_summary_length = self.config.max_summary_length
            self.template_manager.config.max_action_items = self.config.max_action_items
            self.template_manager.config.max_topics = self.config.max_topics
            self.template_manager.config.context_information = self.config.context_information
            self.template_manager.config.custom_instructions = self.config.custom_instructions
            
            self.is_initialized = True
            logger.info("Meeting analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize meeting analyzer: {e}")
            return False
            
    def analyze_meeting(self, transcript: str, metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze a complete meeting transcript"""
        if not self.is_initialized:
            return AnalysisResult(
                analysis_id=str(uuid.uuid4()),
                analysis_type=self.config.analysis_type,
                meeting_type=self.config.meeting_type,
                start_time=datetime.now(),
                end_time=datetime.now(),
                processing_time=0.0,
                success=False,
                error_message="Analyzer not initialized"
            )
            
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting meeting analysis {analysis_id}")
            
            # Create analysis result
            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=self.config.analysis_type,
                meeting_type=self.config.meeting_type,
                start_time=start_time,
                end_time=start_time,  # Will be updated
                processing_time=0.0,
                input_text=transcript,
                input_length=len(transcript)
            )
            
            # Check cache
            cache_key = self._generate_cache_key(transcript, self.config)
            if cache_key in self.analysis_cache:
                logger.info("Using cached analysis result")
                cached_result = self.analysis_cache[cache_key]
                cached_result.analysis_id = analysis_id
                cached_result.start_time = start_time
                return cached_result
                
            # Chunk transcript if needed
            chunks = self._chunk_transcript(transcript)
            result.chunks_processed = len(chunks)
            
            # Perform analysis components
            if self.config.include_summary:
                result.summary = self._generate_summary(chunks)
                
            if self.config.include_action_items:
                result.action_items = self._extract_action_items(chunks)
                
            if self.config.include_topics:
                result.topics = self._identify_topics(chunks)
                
            if self.config.include_participants:
                result.participants = self._analyze_participants(chunks)
                
            if self.config.include_insights:
                result.insights = self._extract_insights(chunks)
                
            if self.config.include_sentiment:
                result.sentiment = self._analyze_sentiment(chunks)
                
            # Calculate quality metrics
            result.confidence = self._calculate_confidence(result)
            result.completeness = self._calculate_completeness(result)
            result.relevance = self._calculate_relevance(result)
            
            # Validate results
            if self.config.require_validation:
                self._validate_results(result)
                
            # Finalize result
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - result.start_time).total_seconds()
            result.success = True
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Meeting analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Meeting analysis {analysis_id} failed: {e}")
            
            result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=self.config.analysis_type,
                meeting_type=self.config.meeting_type,
                start_time=start_time,
                end_time=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds(),
                input_text=transcript,
                input_length=len(transcript),
                success=False,
                error_message=str(e)
            )
            
            self.stats["error_count"] += 1
            return result
            
    def _chunk_transcript(self, transcript: str) -> List[str]:
        """Split transcript into processable chunks"""
        try:
            if len(transcript) <= self.config.chunk_size:
                return [transcript]
                
            chunks = []
            start = 0
            
            while start < len(transcript):
                end = start + self.config.chunk_size
                
                # Find good break point (end of sentence)
                if end < len(transcript):
                    # Look for sentence endings
                    for i in range(end, max(end - 100, start), -1):
                        if transcript[i] in '.!?':
                            end = i + 1
                            break
                            
                chunk = transcript[start:end]
                chunks.append(chunk)
                
                # Move start with overlap
                start = end - self.config.overlap_size
                if start < 0:
                    start = 0
                    
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk transcript: {e}")
            return [transcript]
            
    def _generate_summary(self, chunks: List[str]) -> str:
        """Generate meeting summary from chunks"""
        try:
            # Combine chunks for summary
            combined_text = " ".join(chunks)
            
            # Truncate if too long
            if len(combined_text) > 4000:
                combined_text = combined_text[:4000] + "..."
                
            # Generate prompt
            variables = {
                "transcript": combined_text,
                "output_format": self.config.output_format.value,
                "max_length": self.config.max_summary_length
            }
            
            prompt = self.template_manager.generate_prompt("meeting_summary", variables)
            
            # Generate summary
            result = self.llm_processor.generate(prompt)
            
            if result.success:
                return result.text
            else:
                logger.error(f"Summary generation failed: {result.error_message}")
                return f"Summary generation failed: {result.error_message}"
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation error: {str(e)}"
            
    def _extract_action_items(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Extract action items from chunks"""
        try:
            action_items = []
            
            for i, chunk in enumerate(chunks):
                # Generate prompt for action items
                variables = {
                    "transcript": chunk,
                    "output_format": "structured",
                    "max_items": self.config.max_action_items
                }
                
                prompt = self.template_manager.generate_prompt("action_items", variables)
                
                # Generate action items
                result = self.llm_processor.generate(prompt)
                
                if result.success:
                    # Parse action items (simple parsing for now)
                    chunk_items = self._parse_action_items(result.text)
                    action_items.extend(chunk_items)
                    
            # Deduplicate and limit
            unique_items = self._deduplicate_action_items(action_items)
            return unique_items[:self.config.max_action_items]
            
        except Exception as e:
            logger.error(f"Failed to extract action items: {e}")
            return []
            
    def _identify_topics(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Identify main topics from chunks"""
        try:
            topics = []
            
            # Process chunks to identify topics
            combined_text = " ".join(chunks)
            
            # Truncate if too long
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."
                
            # Generate prompt
            variables = {
                "transcript": combined_text,
                "output_format": "structured",
                "max_topics": self.config.max_topics
            }
            
            prompt = self.template_manager.generate_prompt("topic_identification", variables)
            
            # Generate topics
            result = self.llm_processor.generate(prompt)
            
            if result.success:
                topics = self._parse_topics(result.text)
                
            return topics[:self.config.max_topics]
            
        except Exception as e:
            logger.error(f"Failed to identify topics: {e}")
            return []
            
    def _analyze_participants(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Analyze participant contributions"""
        try:
            participants = []
            
            # Check if transcript has speaker labels
            combined_text = " ".join(chunks)
            if ":" not in combined_text:
                logger.warning("No speaker labels found in transcript")
                return []
                
            # Generate prompt
            variables = {
                "transcript": combined_text,
                "output_format": "structured"
            }
            
            prompt = self.template_manager.generate_prompt("participant_analysis", variables)
            
            # Generate analysis
            result = self.llm_processor.generate(prompt)
            
            if result.success:
                participants = self._parse_participants(result.text)
                
            return participants
            
        except Exception as e:
            logger.error(f"Failed to analyze participants: {e}")
            return []
            
    def _extract_insights(self, chunks: List[str]) -> Dict[str, Any]:
        """Extract meeting insights"""
        try:
            combined_text = " ".join(chunks)
            
            # Generate prompt
            variables = {
                "transcript": combined_text,
                "output_format": "structured"
            }
            
            prompt = self.template_manager.generate_prompt("meeting_insights", variables)
            
            # Generate insights
            result = self.llm_processor.generate(prompt)
            
            if result.success:
                return self._parse_insights(result.text)
            else:
                return {"error": result.error_message}
                
        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")
            return {"error": str(e)}
            
    def _analyze_sentiment(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyze meeting sentiment"""
        try:
            # Simple sentiment analysis placeholder
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "positive_indicators": [],
                "negative_indicators": [],
                "sentiment_trends": []
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {"error": str(e)}
            
    def _parse_action_items(self, text: str) -> List[Dict[str, Any]]:
        """Parse action items from LLM response"""
        try:
            items = []
            
            # Simple parsing - look for numbered items
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Extract action item
                    item = {
                        "description": line,
                        "assignee": "Unknown",
                        "deadline": None,
                        "priority": "Medium",
                        "status": "Open"
                    }
                    items.append(item)
                    
            return items
            
        except Exception as e:
            logger.error(f"Failed to parse action items: {e}")
            return []
            
    def _parse_topics(self, text: str) -> List[Dict[str, Any]]:
        """Parse topics from LLM response"""
        try:
            topics = []
            
            # Simple parsing - look for numbered topics
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Extract topic
                    topic = {
                        "name": line,
                        "importance": "Medium",
                        "time_spent": "Unknown",
                        "participants": [],
                        "outcomes": []
                    }
                    topics.append(topic)
                    
            return topics
            
        except Exception as e:
            logger.error(f"Failed to parse topics: {e}")
            return []
            
    def _parse_participants(self, text: str) -> List[Dict[str, Any]]:
        """Parse participant analysis from LLM response"""
        try:
            participants = []
            
            # Simple parsing - look for participant names
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    name = line.split(':')[0].strip()
                    participant = {
                        "name": name,
                        "participation_level": "Unknown",
                        "contributions": [],
                        "speaking_time": "Unknown"
                    }
                    participants.append(participant)
                    
            return participants
            
        except Exception as e:
            logger.error(f"Failed to parse participants: {e}")
            return []
            
    def _parse_insights(self, text: str) -> Dict[str, Any]:
        """Parse insights from LLM response"""
        try:
            return {
                "key_insights": [text],
                "patterns": [],
                "recommendations": [],
                "risks": [],
                "opportunities": []
            }
            
        except Exception as e:
            logger.error(f"Failed to parse insights: {e}")
            return {"error": str(e)}
            
    def _deduplicate_action_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate action items"""
        try:
            seen = set()
            unique_items = []
            
            for item in items:
                desc = item.get("description", "").lower()
                if desc not in seen:
                    seen.add(desc)
                    unique_items.append(item)
                    
            return unique_items
            
        except Exception as e:
            logger.error(f"Failed to deduplicate action items: {e}")
            return items
            
    def _calculate_confidence(self, result: AnalysisResult) -> float:
        """Calculate overall confidence in analysis"""
        try:
            confidences = []
            
            # Use LLM result confidences
            for llm_result in result.llm_results:
                if llm_result.success:
                    confidences.append(llm_result.confidence)
                    
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.0
            
    def _calculate_completeness(self, result: AnalysisResult) -> float:
        """Calculate analysis completeness"""
        try:
            components = 0
            completed = 0
            
            if self.config.include_summary:
                components += 1
                if result.summary:
                    completed += 1
                    
            if self.config.include_action_items:
                components += 1
                if result.action_items:
                    completed += 1
                    
            if self.config.include_topics:
                components += 1
                if result.topics:
                    completed += 1
                    
            if self.config.include_participants:
                components += 1
                if result.participants:
                    completed += 1
                    
            if self.config.include_insights:
                components += 1
                if result.insights:
                    completed += 1
                    
            if self.config.include_sentiment:
                components += 1
                if result.sentiment:
                    completed += 1
                    
            return completed / components if components > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate completeness: {e}")
            return 0.0
            
    def _calculate_relevance(self, result: AnalysisResult) -> float:
        """Calculate analysis relevance"""
        try:
            # Simple relevance calculation
            if result.input_length > 0:
                content_length = len(result.summary) + len(str(result.action_items)) + len(str(result.topics))
                return min(1.0, content_length / result.input_length)
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance: {e}")
            return 0.0
            
    def _validate_results(self, result: AnalysisResult):
        """Validate analysis results"""
        try:
            # Check minimum confidence
            if result.confidence < self.config.min_confidence:
                result.warnings.append(f"Low confidence: {result.confidence:.2f}")
                
            # Check completeness
            if result.completeness < 0.5:
                result.warnings.append(f"Low completeness: {result.completeness:.2f}")
                
            # Check for empty results
            if not result.summary and not result.action_items and not result.topics:
                result.warnings.append("No substantial analysis results generated")
                
        except Exception as e:
            logger.error(f"Failed to validate results: {e}")
            
    def _generate_cache_key(self, transcript: str, config: AnalysisConfig) -> str:
        """Generate cache key for analysis"""
        try:
            # Use hash of transcript and config
            import hashlib
            content = transcript + str(config.to_dict())
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            return str(uuid.uuid4())
            
    def _update_stats(self, result: AnalysisResult):
        """Update analyzer statistics"""
        try:
            self.stats["total_analyses"] += 1
            
            if result.success:
                self.stats["total_processing_time"] += result.processing_time
                
                # Update average confidence
                count = self.stats["total_analyses"]
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_avg * (count - 1) + result.confidence
                ) / count
                
            # Update success rate
            successes = self.stats["total_analyses"] - self.stats["error_count"]
            self.stats["success_rate"] = successes / self.stats["total_analyses"]
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            "is_initialized": self.is_initialized,
            "config": self.config.to_dict(),
            "stats": self.get_stats(),
            "cache_size": len(self.analysis_cache)
        }
        
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
        
    def shutdown(self):
        """Shutdown analyzer"""
        logger.info("Shutting down meeting analyzer...")
        
        if self.llm_processor:
            self.llm_processor.shutdown()
            
        self.is_initialized = False
        logger.info("Meeting analyzer shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_comprehensive_analyzer() -> MeetingAnalyzer:
    """Create comprehensive meeting analyzer"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.COMPREHENSIVE,
        include_summary=True,
        include_action_items=True,
        include_topics=True,
        include_participants=True,
        include_insights=True,
        include_sentiment=True,
        max_summary_length=1000,
        max_action_items=15,
        max_topics=8
    )
    return MeetingAnalyzer(config)


def create_quick_analyzer() -> MeetingAnalyzer:
    """Create quick meeting analyzer"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.QUICK,
        include_summary=True,
        include_action_items=True,
        include_topics=False,
        include_participants=False,
        include_insights=False,
        include_sentiment=False,
        max_summary_length=300,
        max_action_items=5,
        max_topics=3
    )
    return MeetingAnalyzer(config)


def create_standup_analyzer() -> MeetingAnalyzer:
    """Create standup meeting analyzer"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.SUMMARY,
        meeting_type=MeetingType.STANDUP,
        include_summary=True,
        include_action_items=True,
        include_topics=False,
        include_participants=True,
        include_insights=False,
        include_sentiment=False,
        max_summary_length=200,
        max_action_items=8,
        output_format=OutputFormat.BULLET_POINTS
    )
    return MeetingAnalyzer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Analyzer Test")
    parser.add_argument("--transcript", type=str, required=True, help="Meeting transcript")
    parser.add_argument("--analysis-type", type=str, default="summary",
                       choices=["summary", "quick", "comprehensive"], 
                       help="Analysis type")
    parser.add_argument("--meeting-type", type=str, default="general",
                       choices=["general", "standup", "planning", "review"],
                       help="Meeting type")
    args = parser.parse_args()
    
    # Create analyzer
    if args.analysis_type == "comprehensive":
        analyzer = create_comprehensive_analyzer()
    elif args.analysis_type == "quick":
        analyzer = create_quick_analyzer()
    else:
        analyzer = create_standup_analyzer()
        
    # Update meeting type
    analyzer.config.meeting_type = MeetingType(args.meeting_type)
    
    try:
        # Initialize
        if not analyzer.initialize():
            print("Failed to initialize analyzer")
            sys.exit(1)
            
        print(f"Analyzer status: {analyzer.get_status()}")
        
        # Analyze meeting
        print(f"Analyzing meeting transcript...")
        result = analyzer.analyze_meeting(args.transcript)
        
        if result.success:
            print(f"Analysis completed successfully!")
            print(f"Analysis ID: {result.analysis_id}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Confidence: {result.confidence:.3f}")
            
            if result.summary:
                print(f"\nSummary:\n{result.summary}")
                
            if result.action_items:
                print(f"\nAction Items ({len(result.action_items)}):")
                for i, item in enumerate(result.action_items, 1):
                    print(f"  {i}. {item.get('description', 'Unknown')}")
                    
            if result.topics:
                print(f"\nTopics ({len(result.topics)}):")
                for i, topic in enumerate(result.topics, 1):
                    print(f"  {i}. {topic.get('name', 'Unknown')}")
                    
            if result.participants:
                print(f"\nParticipants ({len(result.participants)}):")
                for participant in result.participants:
                    print(f"  - {participant.get('name', 'Unknown')}")
                    
        else:
            print(f"Analysis failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.shutdown()