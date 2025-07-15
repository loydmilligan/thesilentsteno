#!/usr/bin/env python3
"""
Meeting Analyzer Module

Meeting-specific analysis integration that combines LLM analysis with meeting
context and metadata for comprehensive meeting analysis.

Author: Claude AI Assistant
Date: 2025-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of meeting analysis"""
    QUICK_SUMMARY = "quick_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    CUSTOM = "custom"


@dataclass
class MeetingMetadata:
    """Meeting metadata for analysis context"""
    
    # Basic meeting info
    meeting_id: str
    title: str = ""
    date: Optional[datetime] = None
    duration: float = 0.0  # in seconds
    
    # Participant info
    participants: List[str] = field(default_factory=list)
    organizer: str = ""
    
    # Meeting context
    meeting_type: str = "general"  # general, standup, planning, review, etc.
    agenda: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    
    # Technical metadata
    audio_quality: float = 0.0
    transcription_confidence: float = 0.0
    language: str = "en"
    
    # Custom metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "meeting_id": self.meeting_id,
            "title": self.title,
            "date": self.date.isoformat() if self.date else None,
            "duration": self.duration,
            "participants": self.participants,
            "organizer": self.organizer,
            "meeting_type": self.meeting_type,
            "agenda": self.agenda,
            "goals": self.goals,
            "audio_quality": self.audio_quality,
            "transcription_confidence": self.transcription_confidence,
            "language": self.language,
            "custom_fields": self.custom_fields
        }


@dataclass
class AnalysisConfig:
    """Configuration for meeting analysis"""
    
    # Analysis settings
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    focus_areas: List[str] = field(default_factory=list)
    
    # Content settings
    include_summary: bool = True
    include_action_items: bool = True
    include_decisions: bool = True
    include_topics: bool = True
    include_sentiment: bool = False
    include_participant_analysis: bool = True
    
    # Output settings
    summary_length: int = 500  # words
    max_action_items: int = 20
    max_topics: int = 10
    max_decisions: int = 15
    
    # Quality settings
    min_confidence_threshold: float = 0.7
    require_speaker_attribution: bool = True
    validate_analysis: bool = True
    
    # Context settings
    use_meeting_context: bool = True
    consider_agenda: bool = True
    consider_goals: bool = True
    
    # Custom settings
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    custom_analysis_functions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis_type": self.analysis_type.value,
            "focus_areas": self.focus_areas,
            "include_summary": self.include_summary,
            "include_action_items": self.include_action_items,
            "include_decisions": self.include_decisions,
            "include_topics": self.include_topics,
            "include_sentiment": self.include_sentiment,
            "include_participant_analysis": self.include_participant_analysis,
            "summary_length": self.summary_length,
            "max_action_items": self.max_action_items,
            "max_topics": self.max_topics,
            "max_decisions": self.max_decisions,
            "min_confidence_threshold": self.min_confidence_threshold,
            "require_speaker_attribution": self.require_speaker_attribution,
            "validate_analysis": self.validate_analysis,
            "use_meeting_context": self.use_meeting_context,
            "consider_agenda": self.consider_agenda,
            "consider_goals": self.consider_goals,
            "custom_prompts": self.custom_prompts
        }


@dataclass
class AnalysisResult:
    """Result from meeting analysis"""
    
    # Analysis metadata
    analysis_id: str
    meeting_id: str
    analysis_type: AnalysisType
    timestamp: datetime
    processing_time: float
    
    # Analysis results
    summary: str = ""
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[Dict[str, Any]] = field(default_factory=list)
    key_quotes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Participant analysis
    participant_analysis: Dict[str, Any] = field(default_factory=dict)
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    overall_confidence: float = 0.0
    analysis_quality: float = 0.0
    completeness_score: float = 0.0
    
    # Context and metadata
    meeting_metadata: Optional[MeetingMetadata] = None
    analysis_config: Optional[AnalysisConfig] = None
    
    # Processing info
    llm_results: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis_id": self.analysis_id,
            "meeting_id": self.meeting_id,
            "analysis_type": self.analysis_type.value,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "summary": self.summary,
            "action_items": self.action_items,
            "decisions": self.decisions,
            "topics": self.topics,
            "key_quotes": self.key_quotes,
            "participant_analysis": self.participant_analysis,
            "sentiment_analysis": self.sentiment_analysis,
            "overall_confidence": self.overall_confidence,
            "analysis_quality": self.analysis_quality,
            "completeness_score": self.completeness_score,
            "meeting_metadata": self.meeting_metadata.to_dict() if self.meeting_metadata else None,
            "analysis_config": self.analysis_config.to_dict() if self.analysis_config else None,
            "llm_results": self.llm_results,
            "validation_results": self.validation_results,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings
        }


class MeetingAnalyzer:
    """Meeting-specific analysis integration"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.analyzer_id = str(uuid.uuid4())
        
        # Component references
        self.llm_system = None
        
        # Analysis state
        self.is_initialized = False
        
        # Analysis patterns
        self.action_item_patterns = [
            r"(?i)\b(action|task|todo|follow up|next step|should|will|need to)\b",
            r"(?i)\b(assigned to|responsible for|owned by|take care of)\b",
            r"(?i)\b(by|due|deadline|before|until)\b",
            r"(?i)\b(decide|decision|resolve|figure out|clarify)\b"
        ]
        
        self.decision_patterns = [
            r"(?i)\b(decided|decision|agreed|consensus|conclusion)\b",
            r"(?i)\b(approved|accepted|rejected|denied|postponed)\b",
            r"(?i)\b(vote|voted|unanimous|majority)\b",
            r"(?i)\b(final|finalized|settled|resolved)\b"
        ]
        
        self.topic_patterns = [
            r"(?i)\b(topic|subject|issue|matter|point|question)\b",
            r"(?i)\b(discuss|talk about|address|cover|review)\b",
            r"(?i)\b(regarding|concerning|about|related to)\b"
        ]
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "average_confidence": 0.0,
            "analysis_types": {}
        }
        
        logger.info(f"MeetingAnalyzer initialized with ID: {self.analyzer_id}")
        
    def initialize(self, llm_system=None) -> bool:
        """Initialize the meeting analyzer"""
        try:
            logger.info("Initializing meeting analyzer...")
            
            # Initialize LLM system
            if llm_system:
                self.llm_system = llm_system
            else:
                from .local_llm_processor import LLMAnalysisSystem
                self.llm_system = LLMAnalysisSystem()
                
                if not self.llm_system.initialize():
                    logger.error("Failed to initialize LLM system")
                    return False
                    
            self.is_initialized = True
            logger.info("Meeting analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize meeting analyzer: {e}")
            return False
            
    def analyze_meeting(self, transcript: str, metadata: Optional[MeetingMetadata] = None) -> AnalysisResult:
        """Analyze a meeting with comprehensive analysis"""
        if not self.is_initialized:
            raise Exception("Meeting analyzer not initialized")
            
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        result = AnalysisResult(
            analysis_id=analysis_id,
            meeting_id=metadata.meeting_id if metadata else str(uuid.uuid4()),
            analysis_type=self.config.analysis_type,
            timestamp=start_time,
            processing_time=0.0,
            meeting_metadata=metadata,
            analysis_config=self.config
        )
        
        try:
            logger.info(f"Starting meeting analysis: {analysis_id}")
            
            # Update statistics
            self.stats["total_analyses"] += 1
            analysis_type_key = self.config.analysis_type.value
            self.stats["analysis_types"][analysis_type_key] = self.stats["analysis_types"].get(analysis_type_key, 0) + 1
            
            # Prepare analysis context
            context = self._prepare_analysis_context(transcript, metadata)
            
            # Perform analysis components
            if self.config.include_summary:
                result.summary = self._generate_summary(transcript, context)
                
            if self.config.include_action_items:
                result.action_items = self._extract_action_items(transcript, context)
                
            if self.config.include_decisions:
                result.decisions = self._extract_decisions(transcript, context)
                
            if self.config.include_topics:
                result.topics = self._identify_topics(transcript, context)
                
            if self.config.include_participant_analysis:
                result.participant_analysis = self._analyze_participants(transcript, context)
                
            if self.config.include_sentiment:
                result.sentiment_analysis = self._analyze_sentiment(transcript, context)
                
            # Extract key quotes
            result.key_quotes = self._extract_key_quotes(transcript, context)
            
            # Calculate quality metrics
            result.overall_confidence = self._calculate_overall_confidence(result)
            result.analysis_quality = self._assess_analysis_quality(result)
            result.completeness_score = self._calculate_completeness_score(result)
            
            # Validate analysis if configured
            if self.config.validate_analysis:
                result.validation_results = self._validate_analysis(result)
                
            # Finalize result
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = True
            
            # Update statistics
            self.stats["successful_analyses"] += 1
            self._update_stats(result)
            
            logger.info(f"Meeting analysis completed: {analysis_id}")
            return result
            
        except Exception as e:
            logger.error(f"Meeting analysis failed: {analysis_id}: {e}")
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = False
            result.errors.append(str(e))
            
            self.stats["failed_analyses"] += 1
            return result
            
    def _prepare_analysis_context(self, transcript: str, metadata: Optional[MeetingMetadata]) -> Dict[str, Any]:
        """Prepare context for analysis"""
        context = {
            "transcript_length": len(transcript),
            "word_count": len(transcript.split()),
            "has_speaker_labels": ":" in transcript,
            "language": metadata.language if metadata else "en"
        }
        
        if metadata:
            context.update({
                "meeting_type": metadata.meeting_type,
                "duration": metadata.duration,
                "participant_count": len(metadata.participants),
                "has_agenda": len(metadata.agenda) > 0,
                "has_goals": len(metadata.goals) > 0,
                "agenda": metadata.agenda,
                "goals": metadata.goals
            })
            
        return context
        
    def _generate_summary(self, transcript: str, context: Dict[str, Any]) -> str:
        """Generate meeting summary"""
        try:
            # Prepare prompt for summary generation
            prompt_context = {
                "transcript": transcript,
                "meeting_type": context.get("meeting_type", "general"),
                "word_limit": self.config.summary_length,
                "focus_areas": self.config.focus_areas
            }
            
            if context.get("has_agenda") and self.config.consider_agenda:
                prompt_context["agenda"] = context["agenda"]
                
            if context.get("has_goals") and self.config.consider_goals:
                prompt_context["goals"] = context["goals"]
                
            # Use LLM for summary generation
            summary_result = self.llm_system.generate_summary(
                transcript, 
                template_id="meeting_summary"
            )
            
            if summary_result and summary_result.success:
                return summary_result.text
            else:
                logger.warning("LLM summary generation failed, using fallback")
                return self._generate_fallback_summary(transcript, context)
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._generate_fallback_summary(transcript, context)
            
    def _generate_fallback_summary(self, transcript: str, context: Dict[str, Any]) -> str:
        """Generate fallback summary without LLM"""
        # Simple extractive summary
        sentences = transcript.split('.')
        
        # Get first few sentences and some key sentences
        key_sentences = []
        for i, sentence in enumerate(sentences[:10]):  # First 10 sentences
            if any(keyword in sentence.lower() for keyword in ['important', 'key', 'main', 'summary', 'conclusion']):
                key_sentences.append(sentence.strip())
                
        if not key_sentences:
            key_sentences = sentences[:3]  # First 3 sentences as fallback
            
        return '. '.join(key_sentences) + '.'
        
    def _extract_action_items(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract action items from transcript"""
        try:
            # Use LLM for action item extraction
            action_result = self.llm_system.extract_action_items(transcript)
            
            if action_result and action_result.success:
                return self._format_action_items(action_result.action_items)
            else:
                logger.warning("LLM action extraction failed, using fallback")
                return self._extract_fallback_action_items(transcript, context)
                
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            return self._extract_fallback_action_items(transcript, context)
            
    def _extract_fallback_action_items(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract action items using pattern matching"""
        action_items = []
        
        sentences = transcript.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for action item patterns
            for pattern in self.action_item_patterns:
                if re.search(pattern, sentence):
                    action_items.append({
                        "id": str(uuid.uuid4()),
                        "description": sentence,
                        "assignee": "Unknown",
                        "priority": "medium",
                        "deadline": None,
                        "status": "open",
                        "confidence": 0.6
                    })
                    break
                    
        return action_items[:self.config.max_action_items]
        
    def _format_action_items(self, raw_items: List[Any]) -> List[Dict[str, Any]]:
        """Format action items from LLM output"""
        formatted_items = []
        
        for item in raw_items:
            if hasattr(item, 'to_dict'):
                formatted_items.append(item.to_dict())
            elif isinstance(item, dict):
                formatted_items.append(item)
            else:
                formatted_items.append({
                    "id": str(uuid.uuid4()),
                    "description": str(item),
                    "assignee": "Unknown",
                    "priority": "medium",
                    "deadline": None,
                    "status": "open",
                    "confidence": 0.7
                })
                
        return formatted_items[:self.config.max_action_items]
        
    def _extract_decisions(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract decisions from transcript"""
        try:
            decisions = []
            
            sentences = transcript.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check for decision patterns
                for pattern in self.decision_patterns:
                    if re.search(pattern, sentence):
                        decisions.append({
                            "id": str(uuid.uuid4()),
                            "description": sentence,
                            "type": "decision",
                            "outcome": "approved",  # Default
                            "participants": [],
                            "confidence": 0.7
                        })
                        break
                        
            return decisions[:self.config.max_decisions]
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {e}")
            return []
            
    def _identify_topics(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify main topics from transcript"""
        try:
            # Use LLM for topic identification
            topic_result = self.llm_system.identify_topics(transcript)
            
            if topic_result and topic_result.success:
                return self._format_topics(topic_result.topics)
            else:
                logger.warning("LLM topic identification failed, using fallback")
                return self._identify_fallback_topics(transcript, context)
                
        except Exception as e:
            logger.error(f"Topic identification failed: {e}")
            return self._identify_fallback_topics(transcript, context)
            
    def _identify_fallback_topics(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify topics using pattern matching"""
        topics = []
        
        sentences = transcript.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for topic patterns
            for pattern in self.topic_patterns:
                if re.search(pattern, sentence):
                    topics.append({
                        "id": str(uuid.uuid4()),
                        "name": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                        "description": sentence,
                        "importance": "medium",
                        "relevance_score": 0.7,
                        "mention_count": 1
                    })
                    break
                    
        return topics[:self.config.max_topics]
        
    def _format_topics(self, raw_topics: List[Any]) -> List[Dict[str, Any]]:
        """Format topics from LLM output"""
        formatted_topics = []
        
        for topic in raw_topics:
            if hasattr(topic, 'to_dict'):
                formatted_topics.append(topic.to_dict())
            elif isinstance(topic, dict):
                formatted_topics.append(topic)
            else:
                formatted_topics.append({
                    "id": str(uuid.uuid4()),
                    "name": str(topic),
                    "description": str(topic),
                    "importance": "medium",
                    "relevance_score": 0.7,
                    "mention_count": 1
                })
                
        return formatted_topics[:self.config.max_topics]
        
    def _analyze_participants(self, transcript: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze participant contributions"""
        try:
            if not context.get("has_speaker_labels"):
                return {"note": "No speaker labels found in transcript"}
                
            # Extract speaker segments
            speaker_segments = re.findall(r'(\w+):\s*([^:]+?)(?=\w+:|$)', transcript, re.DOTALL)
            
            participant_stats = {}
            for speaker, content in speaker_segments:
                if speaker not in participant_stats:
                    participant_stats[speaker] = {
                        "name": speaker,
                        "word_count": 0,
                        "speaking_time_percentage": 0.0,
                        "contribution_count": 0,
                        "topics_mentioned": []
                    }
                    
                participant_stats[speaker]["word_count"] += len(content.split())
                participant_stats[speaker]["contribution_count"] += 1
                
            # Calculate speaking time percentages
            total_words = sum(stats["word_count"] for stats in participant_stats.values())
            if total_words > 0:
                for stats in participant_stats.values():
                    stats["speaking_time_percentage"] = (stats["word_count"] / total_words) * 100
                    
            return {
                "total_participants": len(participant_stats),
                "participant_stats": list(participant_stats.values()),
                "most_active_speaker": max(participant_stats.items(), key=lambda x: x[1]["word_count"])[0] if participant_stats else None
            }
            
        except Exception as e:
            logger.error(f"Participant analysis failed: {e}")
            return {"error": str(e)}
            
    def _analyze_sentiment(self, transcript: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of the meeting"""
        try:
            # Simple sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'agree', 'love', 'amazing', 'perfect', 'success']
            negative_words = ['bad', 'terrible', 'disagree', 'hate', 'awful', 'problem', 'issue', 'concern']
            
            words = transcript.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words > 0:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
            else:
                sentiment_score = 0.0
                
            if sentiment_score > 0.2:
                sentiment_label = "positive"
            elif sentiment_score < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
                
            return {
                "overall_sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "confidence": 0.6
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
            
    def _extract_key_quotes(self, transcript: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key quotes from transcript"""
        try:
            quotes = []
            
            # Look for quoted text or emphasized statements
            quoted_patterns = [
                r'"([^"]+)"',  # Direct quotes
                r"'([^']+)'",  # Single quotes
                r'(?i)\b(important|key|main|crucial|critical|essential):\s*([^.!?]+)',  # Important statements
            ]
            
            for pattern in quoted_patterns:
                matches = re.finditer(pattern, transcript)
                for match in matches:
                    quote_text = match.group(1) if match.group(1) else match.group(2)
                    if quote_text and len(quote_text.strip()) > 10:  # Meaningful quotes
                        quotes.append({
                            "id": str(uuid.uuid4()),
                            "text": quote_text.strip(),
                            "speaker": "Unknown",
                            "context": match.group(0),
                            "importance": "medium"
                        })
                        
            return quotes[:10]  # Limit to top 10 quotes
            
        except Exception as e:
            logger.error(f"Key quote extraction failed: {e}")
            return []
            
    def _calculate_overall_confidence(self, result: AnalysisResult) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []
            
            # LLM confidence
            if result.llm_results:
                llm_confidences = [r.get("confidence", 0.0) for r in result.llm_results]
                if llm_confidences:
                    confidence_factors.append(sum(llm_confidences) / len(llm_confidences))
                    
            # Content completeness
            if result.summary:
                confidence_factors.append(0.8)
            if result.action_items:
                confidence_factors.append(0.7)
            if result.topics:
                confidence_factors.append(0.6)
                
            # Transcript quality
            if result.meeting_metadata:
                confidence_factors.append(result.meeting_metadata.transcription_confidence)
                
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
            
    def _assess_analysis_quality(self, result: AnalysisResult) -> float:
        """Assess the quality of the analysis"""
        try:
            quality_factors = []
            
            # Content quality
            if result.summary and len(result.summary) > 50:
                quality_factors.append(0.8)
            if result.action_items and len(result.action_items) > 0:
                quality_factors.append(0.7)
            if result.topics and len(result.topics) > 0:
                quality_factors.append(0.6)
                
            # Completeness
            expected_components = 0
            actual_components = 0
            
            if self.config.include_summary:
                expected_components += 1
                if result.summary:
                    actual_components += 1
                    
            if self.config.include_action_items:
                expected_components += 1
                if result.action_items:
                    actual_components += 1
                    
            if self.config.include_topics:
                expected_components += 1
                if result.topics:
                    actual_components += 1
                    
            if expected_components > 0:
                completeness = actual_components / expected_components
                quality_factors.append(completeness)
                
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0
            
    def _calculate_completeness_score(self, result: AnalysisResult) -> float:
        """Calculate completeness score"""
        try:
            total_components = 0
            completed_components = 0
            
            components = [
                ("summary", result.summary),
                ("action_items", result.action_items),
                ("decisions", result.decisions),
                ("topics", result.topics),
                ("participant_analysis", result.participant_analysis),
                ("sentiment_analysis", result.sentiment_analysis)
            ]
            
            for component_name, component_value in components:
                total_components += 1
                if component_value:
                    completed_components += 1
                    
            return completed_components / total_components if total_components > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 0.0
            
    def _validate_analysis(self, result: AnalysisResult) -> Dict[str, Any]:
        """Validate analysis results"""
        try:
            validation = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
            
            # Check minimum confidence threshold
            if result.overall_confidence < self.config.min_confidence_threshold:
                validation["issues"].append(f"Overall confidence ({result.overall_confidence:.2f}) below threshold ({self.config.min_confidence_threshold})")
                validation["is_valid"] = False
                
            # Check required components
            if self.config.include_summary and not result.summary:
                validation["issues"].append("Summary is required but missing")
                validation["is_valid"] = False
                
            # Check action items quality
            if result.action_items:
                for item in result.action_items:
                    if not item.get("description"):
                        validation["warnings"].append("Action item missing description")
                        
            # Check topics quality
            if result.topics:
                for topic in result.topics:
                    if not topic.get("name"):
                        validation["warnings"].append("Topic missing name")
                        
            return validation
            
        except Exception as e:
            logger.error(f"Analysis validation failed: {e}")
            return {"is_valid": False, "issues": [str(e)]}
            
    def _update_stats(self, result: AnalysisResult):
        """Update analyzer statistics"""
        try:
            # Update average processing time
            total = self.stats["total_analyses"]
            old_avg = self.stats["average_processing_time"]
            self.stats["average_processing_time"] = (
                old_avg * (total - 1) + result.processing_time
            ) / total
            
            # Update average confidence
            old_conf_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (
                old_conf_avg * (total - 1) + result.overall_confidence
            ) / total
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            "analyzer_id": self.analyzer_id,
            "is_initialized": self.is_initialized,
            "config": self.config.to_dict(),
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown analyzer"""
        logger.info("Shutting down meeting analyzer...")
        
        if self.llm_system:
            self.llm_system.shutdown()
            
        self.is_initialized = False
        logger.info("Meeting analyzer shutdown complete")


# Factory functions
def create_quick_analyzer() -> MeetingAnalyzer:
    """Create quick analyzer for fast analysis"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.QUICK_SUMMARY,
        include_summary=True,
        include_action_items=True,
        include_decisions=False,
        include_topics=False,
        include_sentiment=False,
        include_participant_analysis=False,
        summary_length=200,
        max_action_items=5
    )
    return MeetingAnalyzer(config)


def create_comprehensive_analyzer() -> MeetingAnalyzer:
    """Create comprehensive analyzer for detailed analysis"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.COMPREHENSIVE,
        include_summary=True,
        include_action_items=True,
        include_decisions=True,
        include_topics=True,
        include_sentiment=True,
        include_participant_analysis=True,
        summary_length=500,
        max_action_items=20,
        max_topics=10,
        max_decisions=15
    )
    return MeetingAnalyzer(config)


def create_focused_analyzer(focus_areas: List[str]) -> MeetingAnalyzer:
    """Create focused analyzer for specific areas"""
    config = AnalysisConfig(
        analysis_type=AnalysisType.FOCUSED,
        focus_areas=focus_areas,
        include_summary=True,
        include_action_items=True,
        include_decisions=True,
        include_topics=True,
        include_sentiment=False,
        include_participant_analysis=True
    )
    return MeetingAnalyzer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Analyzer Test")
    parser.add_argument("--transcript", type=str, required=True, help="Meeting transcript")
    parser.add_argument("--analyzer-type", type=str, default="comprehensive",
                       choices=["quick", "comprehensive", "focused"], help="Analyzer type")
    parser.add_argument("--focus-areas", type=str, nargs="*", default=[], help="Focus areas for focused analyzer")
    args = parser.parse_args()
    
    # Create analyzer
    if args.analyzer_type == "quick":
        analyzer = create_quick_analyzer()
    elif args.analyzer_type == "focused":
        analyzer = create_focused_analyzer(args.focus_areas)
    else:
        analyzer = create_comprehensive_analyzer()
        
    try:
        print(f"Analyzer status: {analyzer.get_status()}")
        
        # Initialize analyzer
        if not analyzer.initialize():
            print("Failed to initialize analyzer")
            sys.exit(1)
            
        # Create sample metadata
        metadata = MeetingMetadata(
            meeting_id="test_meeting_001",
            title="Test Meeting",
            date=datetime.now(),
            meeting_type="general",
            participants=["Alice", "Bob", "Charlie"]
        )
        
        # Analyze meeting
        print("Analyzing meeting...")
        result = analyzer.analyze_meeting(args.transcript, metadata)
        
        if result.success:
            print(f"Analysis completed successfully!")
            print(f"Analysis ID: {result.analysis_id}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Overall confidence: {result.overall_confidence:.3f}")
            
            if result.summary:
                print(f"\nSummary:\n{result.summary}")
                
            if result.action_items:
                print(f"\nAction Items ({len(result.action_items)}):")
                for i, item in enumerate(result.action_items, 1):
                    print(f"  {i}. {item['description']}")
                    
            if result.topics:
                print(f"\nTopics ({len(result.topics)}):")
                for i, topic in enumerate(result.topics, 1):
                    print(f"  {i}. {topic['name']}")
                    
        else:
            print(f"Analysis failed: {result.errors}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.shutdown()