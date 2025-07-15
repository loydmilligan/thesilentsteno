#!/usr/bin/env python3
"""
Participant Analyzer Module

Analyzes participant speaking patterns, engagement metrics, and contribution
statistics for meeting transcripts with speaker diarization.

Author: Claude AI Assistant
Date: 2025-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngagementLevel(Enum):
    """Participant engagement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ParticipationPattern(Enum):
    """Participation patterns"""
    DOMINANT = "dominant"
    BALANCED = "balanced"
    QUIET = "quiet"
    INTERRUPTED = "interrupted"
    FACILITATOR = "facilitator"
    RESPONSIVE = "responsive"


@dataclass
class SpeakingPattern:
    """Speaking pattern analysis for a participant"""
    
    # Basic metrics
    total_words: int = 0
    speaking_time_seconds: float = 0.0
    turn_count: int = 0
    
    # Timing patterns
    average_turn_length: float = 0.0
    longest_turn_length: float = 0.0
    shortest_turn_length: float = 0.0
    
    # Interaction patterns
    interruptions_made: int = 0
    interruptions_received: int = 0
    questions_asked: int = 0
    responses_given: int = 0
    
    # Content patterns
    topic_initiations: int = 0
    agreements: int = 0
    disagreements: int = 0
    
    # Temporal patterns
    speaking_distribution: List[float] = field(default_factory=list)  # Speaking time per time segment
    participation_consistency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_words": self.total_words,
            "speaking_time_seconds": self.speaking_time_seconds,
            "turn_count": self.turn_count,
            "average_turn_length": self.average_turn_length,
            "longest_turn_length": self.longest_turn_length,
            "shortest_turn_length": self.shortest_turn_length,
            "interruptions_made": self.interruptions_made,
            "interruptions_received": self.interruptions_received,
            "questions_asked": self.questions_asked,
            "responses_given": self.responses_given,
            "topic_initiations": self.topic_initiations,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "speaking_distribution": self.speaking_distribution,
            "participation_consistency": self.participation_consistency
        }


@dataclass
class EngagementMetrics:
    """Engagement metrics for a participant"""
    
    # Overall engagement
    engagement_level: EngagementLevel = EngagementLevel.MEDIUM
    engagement_score: float = 0.0
    
    # Participation quality
    contribution_quality: float = 0.0
    interaction_quality: float = 0.0
    
    # Behavioral indicators
    proactive_participation: float = 0.0
    collaborative_behavior: float = 0.0
    leadership_indicators: float = 0.0
    
    # Attention indicators
    responsiveness: float = 0.0
    topic_relevance: float = 0.0
    
    # Communication style
    communication_style: str = "neutral"
    assertiveness: float = 0.0
    supportiveness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "engagement_level": self.engagement_level.value,
            "engagement_score": self.engagement_score,
            "contribution_quality": self.contribution_quality,
            "interaction_quality": self.interaction_quality,
            "proactive_participation": self.proactive_participation,
            "collaborative_behavior": self.collaborative_behavior,
            "leadership_indicators": self.leadership_indicators,
            "responsiveness": self.responsiveness,
            "topic_relevance": self.topic_relevance,
            "communication_style": self.communication_style,
            "assertiveness": self.assertiveness,
            "supportiveness": self.supportiveness
        }


@dataclass
class ParticipantStats:
    """Comprehensive participant statistics"""
    
    # Identity
    participant_id: str
    name: str
    
    # Basic metrics
    speaking_time_percentage: float = 0.0
    word_count: int = 0
    turn_count: int = 0
    
    # Patterns and engagement
    speaking_pattern: SpeakingPattern = field(default_factory=SpeakingPattern)
    engagement_metrics: EngagementMetrics = field(default_factory=EngagementMetrics)
    participation_pattern: ParticipationPattern = ParticipationPattern.BALANCED
    
    # Interaction data
    interactions_with: Dict[str, int] = field(default_factory=dict)
    topics_contributed_to: List[str] = field(default_factory=list)
    
    # Quality indicators
    confidence_score: float = 0.0
    analysis_quality: float = 0.0
    
    # Temporal data
    first_contribution_time: float = 0.0
    last_contribution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "speaking_time_percentage": self.speaking_time_percentage,
            "word_count": self.word_count,
            "turn_count": self.turn_count,
            "speaking_pattern": self.speaking_pattern.to_dict(),
            "engagement_metrics": self.engagement_metrics.to_dict(),
            "participation_pattern": self.participation_pattern.value,
            "interactions_with": self.interactions_with,
            "topics_contributed_to": self.topics_contributed_to,
            "confidence_score": self.confidence_score,
            "analysis_quality": self.analysis_quality,
            "first_contribution_time": self.first_contribution_time,
            "last_contribution_time": self.last_contribution_time
        }


@dataclass
class ParticipantConfig:
    """Configuration for participant analysis"""
    
    # Analysis settings
    min_word_threshold: int = 10
    time_segment_duration: int = 30  # seconds
    
    # Pattern detection
    detect_interruptions: bool = True
    detect_questions: bool = True
    detect_agreements: bool = True
    detect_topic_initiations: bool = True
    
    # Engagement calculation
    engagement_factors: Dict[str, float] = field(default_factory=lambda: {
        "speaking_time": 0.3,
        "interaction_quality": 0.2,
        "topic_relevance": 0.2,
        "proactive_participation": 0.15,
        "collaborative_behavior": 0.15
    })
    
    # Quality settings
    min_confidence_threshold: float = 0.6
    require_speaker_labels: bool = True
    
    # Output settings
    include_temporal_analysis: bool = True
    include_interaction_matrix: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "min_word_threshold": self.min_word_threshold,
            "time_segment_duration": self.time_segment_duration,
            "detect_interruptions": self.detect_interruptions,
            "detect_questions": self.detect_questions,
            "detect_agreements": self.detect_agreements,
            "detect_topic_initiations": self.detect_topic_initiations,
            "engagement_factors": self.engagement_factors,
            "min_confidence_threshold": self.min_confidence_threshold,
            "require_speaker_labels": self.require_speaker_labels,
            "include_temporal_analysis": self.include_temporal_analysis,
            "include_interaction_matrix": self.include_interaction_matrix
        }


@dataclass
class ParticipantAnalysisResult:
    """Result from participant analysis"""
    
    # Analysis metadata
    analysis_id: str
    timestamp: datetime
    processing_time: float
    
    # Participants
    participant_stats: List[ParticipantStats] = field(default_factory=list)
    
    # Group dynamics
    total_participants: int = 0
    most_active_participant: str = ""
    least_active_participant: str = ""
    dominant_speakers: List[str] = field(default_factory=list)
    
    # Interaction analysis
    interaction_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    collaboration_score: float = 0.0
    
    # Temporal analysis
    participation_timeline: List[Dict[str, Any]] = field(default_factory=list)
    engagement_trends: Dict[str, List[float]] = field(default_factory=dict)
    
    # Quality metrics
    overall_engagement: float = 0.0
    participation_balance: float = 0.0
    analysis_confidence: float = 0.0
    
    # Status
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "participant_stats": [p.to_dict() for p in self.participant_stats],
            "total_participants": self.total_participants,
            "most_active_participant": self.most_active_participant,
            "least_active_participant": self.least_active_participant,
            "dominant_speakers": self.dominant_speakers,
            "interaction_matrix": self.interaction_matrix,
            "collaboration_score": self.collaboration_score,
            "participation_timeline": self.participation_timeline,
            "engagement_trends": self.engagement_trends,
            "overall_engagement": self.overall_engagement,
            "participation_balance": self.participation_balance,
            "analysis_confidence": self.analysis_confidence,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings
        }


class ParticipantAnalyzer:
    """Participant analysis system"""
    
    def __init__(self, config: Optional[ParticipantConfig] = None):
        self.config = config or ParticipantConfig()
        self.analyzer_id = str(uuid.uuid4())
        
        # Pattern detection
        self.question_patterns = [
            r'\?',
            r'(?i)\b(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)\b.*\?',
            r'(?i)\b(tell me|explain|clarify|elaborate)\b'
        ]
        
        self.agreement_patterns = [
            r'(?i)\b(agree|yes|correct|right|exactly|definitely|absolutely|sure|okay|ok)\b',
            r'(?i)\b(i think so|that\'s right|good point|makes sense)\b'
        ]
        
        self.disagreement_patterns = [
            r'(?i)\b(disagree|no|wrong|incorrect|false|doubt|question)\b',
            r'(?i)\b(i don\'t think|not sure|but|however|actually)\b'
        ]
        
        self.topic_initiation_patterns = [
            r'(?i)\b(let\'s talk about|next topic|moving on|i want to discuss)\b',
            r'(?i)\b(another thing|also|additionally|furthermore)\b'
        ]
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "average_participant_count": 0.0,
            "average_confidence": 0.0
        }
        
        logger.info(f"ParticipantAnalyzer initialized with ID: {self.analyzer_id}")
        
    def analyze_participants(self, transcript: str, 
                           speaker_data: Optional[Dict[str, Any]] = None) -> ParticipantAnalysisResult:
        """Analyze participants from transcript"""
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        result = ParticipantAnalysisResult(
            analysis_id=analysis_id,
            timestamp=start_time,
            processing_time=0.0
        )
        
        try:
            logger.info(f"Starting participant analysis: {analysis_id}")
            
            # Update statistics
            self.stats["total_analyses"] += 1
            
            # Extract speaker segments
            speaker_segments = self._extract_speaker_segments(transcript)
            
            if not speaker_segments:
                if self.config.require_speaker_labels:
                    raise ValueError("No speaker labels found in transcript")
                else:
                    result.warnings.append("No speaker labels found, limited analysis available")
                    return result
                    
            # Analyze each participant
            participant_stats = []
            for speaker_name in set(seg['speaker'] for seg in speaker_segments):
                stats = self._analyze_participant(speaker_name, speaker_segments, transcript)
                participant_stats.append(stats)
                
            result.participant_stats = participant_stats
            result.total_participants = len(participant_stats)
            
            # Calculate group dynamics
            self._calculate_group_dynamics(result)
            
            # Analyze interactions
            if self.config.include_interaction_matrix:
                result.interaction_matrix = self._analyze_interactions(speaker_segments)
                
            # Temporal analysis
            if self.config.include_temporal_analysis:
                result.participation_timeline = self._analyze_temporal_patterns(speaker_segments)
                result.engagement_trends = self._calculate_engagement_trends(speaker_segments)
                
            # Calculate overall metrics
            result.overall_engagement = self._calculate_overall_engagement(participant_stats)
            result.participation_balance = self._calculate_participation_balance(participant_stats)
            result.collaboration_score = self._calculate_collaboration_score(result.interaction_matrix)
            result.analysis_confidence = self._calculate_analysis_confidence(result)
            
            # Finalize result
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = True
            
            # Update statistics
            self.stats["successful_analyses"] += 1
            self._update_stats(result)
            
            logger.info(f"Participant analysis completed: {analysis_id}")
            return result
            
        except Exception as e:
            logger.error(f"Participant analysis failed: {analysis_id}: {e}")
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = False
            result.errors.append(str(e))
            
            self.stats["failed_analyses"] += 1
            return result
            
    def _extract_speaker_segments(self, transcript: str) -> List[Dict[str, Any]]:
        """Extract speaker segments from transcript"""
        try:
            segments = []
            
            # Find all speaker segments
            speaker_pattern = r'(\w+):\s*([^:]+?)(?=\w+:|$)'
            matches = re.finditer(speaker_pattern, transcript, re.DOTALL)
            
            for match in matches:
                speaker = match.group(1)
                content = match.group(2).strip()
                
                if content:  # Only include non-empty segments
                    segments.append({
                        'speaker': speaker,
                        'content': content,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'word_count': len(content.split()),
                        'char_count': len(content)
                    })
                    
            return segments
            
        except Exception as e:
            logger.error(f"Failed to extract speaker segments: {e}")
            return []
            
    def _analyze_participant(self, speaker_name: str, 
                           speaker_segments: List[Dict[str, Any]], 
                           full_transcript: str) -> ParticipantStats:
        """Analyze individual participant"""
        try:
            # Get all segments for this speaker
            participant_segments = [seg for seg in speaker_segments if seg['speaker'] == speaker_name]
            
            if not participant_segments:
                return ParticipantStats(
                    participant_id=str(uuid.uuid4()),
                    name=speaker_name
                )
                
            # Create participant stats
            stats = ParticipantStats(
                participant_id=str(uuid.uuid4()),
                name=speaker_name
            )
            
            # Basic metrics
            stats.word_count = sum(seg['word_count'] for seg in participant_segments)
            stats.turn_count = len(participant_segments)
            
            # Calculate speaking time (approximate)
            total_chars = sum(seg['char_count'] for seg in participant_segments)
            stats.speaking_time_percentage = (total_chars / len(full_transcript)) * 100
            
            # Analyze speaking patterns
            stats.speaking_pattern = self._analyze_speaking_pattern(participant_segments)
            
            # Analyze engagement
            stats.engagement_metrics = self._analyze_engagement(participant_segments, speaker_segments)
            
            # Determine participation pattern
            stats.participation_pattern = self._determine_participation_pattern(stats)
            
            # Analyze interactions
            stats.interactions_with = self._analyze_participant_interactions(
                speaker_name, speaker_segments
            )
            
            # Calculate confidence and quality
            stats.confidence_score = self._calculate_participant_confidence(stats)
            stats.analysis_quality = self._assess_participant_analysis_quality(stats)
            
            # Temporal data
            if participant_segments:
                stats.first_contribution_time = participant_segments[0]['start_pos']
                stats.last_contribution_time = participant_segments[-1]['end_pos']
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to analyze participant {speaker_name}: {e}")
            return ParticipantStats(
                participant_id=str(uuid.uuid4()),
                name=speaker_name
            )
            
    def _analyze_speaking_pattern(self, segments: List[Dict[str, Any]]) -> SpeakingPattern:
        """Analyze speaking patterns for a participant"""
        try:
            pattern = SpeakingPattern()
            
            if not segments:
                return pattern
                
            # Basic metrics
            pattern.total_words = sum(seg['word_count'] for seg in segments)
            pattern.turn_count = len(segments)
            
            # Turn length analysis
            turn_lengths = [seg['word_count'] for seg in segments]
            pattern.average_turn_length = sum(turn_lengths) / len(turn_lengths)
            pattern.longest_turn_length = max(turn_lengths)
            pattern.shortest_turn_length = min(turn_lengths)
            
            # Analyze content patterns
            all_content = ' '.join(seg['content'] for seg in segments)
            
            # Count questions
            if self.config.detect_questions:
                pattern.questions_asked = self._count_patterns(all_content, self.question_patterns)
                
            # Count agreements/disagreements
            if self.config.detect_agreements:
                pattern.agreements = self._count_patterns(all_content, self.agreement_patterns)
                pattern.disagreements = self._count_patterns(all_content, self.disagreement_patterns)
                
            # Count topic initiations
            if self.config.detect_topic_initiations:
                pattern.topic_initiations = self._count_patterns(all_content, self.topic_initiation_patterns)
                
            # Calculate consistency (how evenly distributed the speaking is)
            if len(segments) > 1:
                word_counts = [seg['word_count'] for seg in segments]
                mean_words = sum(word_counts) / len(word_counts)
                variance = sum((count - mean_words) ** 2 for count in word_counts) / len(word_counts)
                pattern.participation_consistency = 1.0 / (1.0 + variance / mean_words) if mean_words > 0 else 0.0
                
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to analyze speaking pattern: {e}")
            return SpeakingPattern()
            
    def _analyze_engagement(self, participant_segments: List[Dict[str, Any]], 
                          all_segments: List[Dict[str, Any]]) -> EngagementMetrics:
        """Analyze engagement metrics for a participant"""
        try:
            metrics = EngagementMetrics()
            
            if not participant_segments:
                return metrics
                
            # Calculate basic engagement score
            word_count = sum(seg['word_count'] for seg in participant_segments)
            total_words = sum(seg['word_count'] for seg in all_segments)
            
            if total_words > 0:
                speaking_ratio = word_count / total_words
                
                # Normalize to 0-1 scale (assuming balanced participation is around 1/n participants)
                expected_ratio = 1.0 / len(set(seg['speaker'] for seg in all_segments))
                normalized_ratio = min(1.0, speaking_ratio / expected_ratio)
                
                metrics.engagement_score = normalized_ratio * 0.5  # Base score from speaking time
                
            # Analyze content for engagement indicators
            all_content = ' '.join(seg['content'] for seg in participant_segments)
            
            # Proactive participation (questions, topic initiations)
            proactive_indicators = (
                self._count_patterns(all_content, self.question_patterns) +
                self._count_patterns(all_content, self.topic_initiation_patterns)
            )
            metrics.proactive_participation = min(1.0, proactive_indicators / 10.0)
            
            # Collaborative behavior (agreements, supportive language)
            collaborative_indicators = self._count_patterns(all_content, self.agreement_patterns)
            metrics.collaborative_behavior = min(1.0, collaborative_indicators / 5.0)
            
            # Responsiveness (responses to others)
            # This is a simplified version - in practice, would need more sophisticated analysis
            metrics.responsiveness = 0.5  # Placeholder
            
            # Topic relevance (simplified)
            metrics.topic_relevance = 0.7  # Placeholder
            
            # Calculate overall engagement score
            factors = self.config.engagement_factors
            metrics.engagement_score = (
                factors["speaking_time"] * normalized_ratio +
                factors["proactive_participation"] * metrics.proactive_participation +
                factors["collaborative_behavior"] * metrics.collaborative_behavior +
                factors["interaction_quality"] * metrics.responsiveness +
                factors["topic_relevance"] * metrics.topic_relevance
            )
            
            # Determine engagement level
            if metrics.engagement_score > 0.8:
                metrics.engagement_level = EngagementLevel.VERY_HIGH
            elif metrics.engagement_score > 0.6:
                metrics.engagement_level = EngagementLevel.HIGH
            elif metrics.engagement_score > 0.4:
                metrics.engagement_level = EngagementLevel.MEDIUM
            else:
                metrics.engagement_level = EngagementLevel.LOW
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze engagement: {e}")
            return EngagementMetrics()
            
    def _determine_participation_pattern(self, stats: ParticipantStats) -> ParticipationPattern:
        """Determine participation pattern for a participant"""
        try:
            # Simple heuristics for pattern determination
            if stats.speaking_time_percentage > 40:
                return ParticipationPattern.DOMINANT
            elif stats.speaking_time_percentage < 10:
                return ParticipationPattern.QUIET
            elif stats.speaking_pattern.questions_asked > 5:
                return ParticipationPattern.FACILITATOR
            elif stats.speaking_pattern.agreements > stats.speaking_pattern.disagreements * 2:
                return ParticipationPattern.RESPONSIVE
            else:
                return ParticipationPattern.BALANCED
                
        except Exception as e:
            logger.error(f"Failed to determine participation pattern: {e}")
            return ParticipationPattern.BALANCED
            
    def _analyze_participant_interactions(self, speaker_name: str, 
                                        all_segments: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze interactions between participants"""
        try:
            interactions = defaultdict(int)
            
            # Find segments where this speaker follows another speaker
            for i, segment in enumerate(all_segments):
                if segment['speaker'] == speaker_name and i > 0:
                    previous_speaker = all_segments[i-1]['speaker']
                    if previous_speaker != speaker_name:
                        interactions[previous_speaker] += 1
                        
            return dict(interactions)
            
        except Exception as e:
            logger.error(f"Failed to analyze interactions for {speaker_name}: {e}")
            return {}
            
    def _calculate_group_dynamics(self, result: ParticipantAnalysisResult):
        """Calculate group dynamics metrics"""
        try:
            if not result.participant_stats:
                return
                
            # Find most and least active participants
            most_active = max(result.participant_stats, key=lambda p: p.speaking_time_percentage)
            least_active = min(result.participant_stats, key=lambda p: p.speaking_time_percentage)
            
            result.most_active_participant = most_active.name
            result.least_active_participant = least_active.name
            
            # Find dominant speakers (above average + 1 std dev)
            speaking_times = [p.speaking_time_percentage for p in result.participant_stats]
            mean_speaking_time = sum(speaking_times) / len(speaking_times)
            
            if len(speaking_times) > 1:
                variance = sum((t - mean_speaking_time) ** 2 for t in speaking_times) / len(speaking_times)
                std_dev = math.sqrt(variance)
                threshold = mean_speaking_time + std_dev
                
                result.dominant_speakers = [
                    p.name for p in result.participant_stats 
                    if p.speaking_time_percentage > threshold
                ]
                
        except Exception as e:
            logger.error(f"Failed to calculate group dynamics: {e}")
            
    def _analyze_interactions(self, speaker_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Analyze interaction matrix between participants"""
        try:
            matrix = defaultdict(lambda: defaultdict(int))
            
            # Build interaction matrix
            for i in range(1, len(speaker_segments)):
                current_speaker = speaker_segments[i]['speaker']
                previous_speaker = speaker_segments[i-1]['speaker']
                
                if current_speaker != previous_speaker:
                    matrix[previous_speaker][current_speaker] += 1
                    
            # Convert to regular dict
            return {k: dict(v) for k, v in matrix.items()}
            
        except Exception as e:
            logger.error(f"Failed to analyze interactions: {e}")
            return {}
            
    def _analyze_temporal_patterns(self, speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal participation patterns"""
        try:
            timeline = []
            
            # Create timeline events
            for segment in speaker_segments:
                timeline.append({
                    'timestamp': segment['start_pos'],
                    'speaker': segment['speaker'],
                    'word_count': segment['word_count'],
                    'content_preview': segment['content'][:100] + '...' if len(segment['content']) > 100 else segment['content']
                })
                
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal patterns: {e}")
            return []
            
    def _calculate_engagement_trends(self, speaker_segments: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Calculate engagement trends over time"""
        try:
            trends = defaultdict(list)
            
            # Group segments by speaker
            speaker_segments_map = defaultdict(list)
            for segment in speaker_segments:
                speaker_segments_map[segment['speaker']].append(segment)
                
            # Calculate engagement trend for each speaker
            for speaker, segments in speaker_segments_map.items():
                segment_engagement = []
                for segment in segments:
                    # Simple engagement metric based on word count
                    engagement = min(1.0, segment['word_count'] / 50.0)
                    segment_engagement.append(engagement)
                    
                trends[speaker] = segment_engagement
                
            return dict(trends)
            
        except Exception as e:
            logger.error(f"Failed to calculate engagement trends: {e}")
            return {}
            
    def _calculate_overall_engagement(self, participant_stats: List[ParticipantStats]) -> float:
        """Calculate overall engagement score"""
        try:
            if not participant_stats:
                return 0.0
                
            engagement_scores = [p.engagement_metrics.engagement_score for p in participant_stats]
            return sum(engagement_scores) / len(engagement_scores)
            
        except Exception as e:
            logger.error(f"Failed to calculate overall engagement: {e}")
            return 0.0
            
    def _calculate_participation_balance(self, participant_stats: List[ParticipantStats]) -> float:
        """Calculate participation balance score"""
        try:
            if not participant_stats:
                return 0.0
                
            speaking_times = [p.speaking_time_percentage for p in participant_stats]
            
            # Calculate coefficient of variation (lower is more balanced)
            mean_time = sum(speaking_times) / len(speaking_times)
            if mean_time == 0:
                return 0.0
                
            variance = sum((t - mean_time) ** 2 for t in speaking_times) / len(speaking_times)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_time
            
            # Convert to balance score (0-1, higher is more balanced)
            balance_score = 1.0 / (1.0 + cv)
            return balance_score
            
        except Exception as e:
            logger.error(f"Failed to calculate participation balance: {e}")
            return 0.0
            
    def _calculate_collaboration_score(self, interaction_matrix: Dict[str, Dict[str, int]]) -> float:
        """Calculate collaboration score based on interactions"""
        try:
            if not interaction_matrix:
                return 0.0
                
            # Count total interactions
            total_interactions = sum(
                sum(interactions.values()) for interactions in interaction_matrix.values()
            )
            
            # Count unique interaction pairs
            unique_pairs = set()
            for speaker, interactions in interaction_matrix.items():
                for other_speaker in interactions:
                    pair = tuple(sorted([speaker, other_speaker]))
                    unique_pairs.add(pair)
                    
            # Calculate collaboration score
            if len(unique_pairs) > 0:
                return min(1.0, total_interactions / (len(unique_pairs) * 5))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate collaboration score: {e}")
            return 0.0
            
    def _calculate_analysis_confidence(self, result: ParticipantAnalysisResult) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence_factors = []
            
            # Confidence based on data availability
            if result.participant_stats:
                avg_confidence = sum(p.confidence_score for p in result.participant_stats) / len(result.participant_stats)
                confidence_factors.append(avg_confidence)
                
            # Confidence based on participant count
            if result.total_participants > 1:
                confidence_factors.append(min(1.0, result.total_participants / 5.0))
                
            # Confidence based on interaction data
            if result.interaction_matrix:
                confidence_factors.append(0.8)
                
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate analysis confidence: {e}")
            return 0.0
            
    def _calculate_participant_confidence(self, stats: ParticipantStats) -> float:
        """Calculate confidence score for individual participant"""
        try:
            confidence_factors = []
            
            # Confidence based on data volume
            if stats.word_count >= self.config.min_word_threshold:
                confidence_factors.append(min(1.0, stats.word_count / 100.0))
                
            # Confidence based on turn count
            if stats.turn_count > 0:
                confidence_factors.append(min(1.0, stats.turn_count / 10.0))
                
            # Confidence based on engagement metrics
            if stats.engagement_metrics.engagement_score > 0:
                confidence_factors.append(stats.engagement_metrics.engagement_score)
                
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate participant confidence: {e}")
            return 0.0
            
    def _assess_participant_analysis_quality(self, stats: ParticipantStats) -> float:
        """Assess the quality of participant analysis"""
        try:
            quality_factors = []
            
            # Quality based on completeness
            if stats.speaking_pattern.total_words > 0:
                quality_factors.append(0.8)
            if stats.engagement_metrics.engagement_score > 0:
                quality_factors.append(0.7)
            if stats.interactions_with:
                quality_factors.append(0.6)
                
            # Quality based on consistency
            if stats.speaking_pattern.participation_consistency > 0.5:
                quality_factors.append(0.5)
                
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            logger.error(f"Failed to assess analysis quality: {e}")
            return 0.0
            
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count occurrences of patterns in text"""
        try:
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text)
                count += len(matches)
            return count
            
        except Exception as e:
            logger.error(f"Failed to count patterns: {e}")
            return 0
            
    def _update_stats(self, result: ParticipantAnalysisResult):
        """Update analyzer statistics"""
        try:
            # Update average processing time
            total = self.stats["total_analyses"]
            old_avg = self.stats["average_processing_time"]
            self.stats["average_processing_time"] = (
                old_avg * (total - 1) + result.processing_time
            ) / total
            
            # Update average participant count
            old_part_avg = self.stats["average_participant_count"]
            self.stats["average_participant_count"] = (
                old_part_avg * (total - 1) + result.total_participants
            ) / total
            
            # Update average confidence
            old_conf_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (
                old_conf_avg * (total - 1) + result.analysis_confidence
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
            "config": self.config.to_dict(),
            "stats": self.get_stats()
        }


# Factory functions
def create_basic_participant_analyzer() -> ParticipantAnalyzer:
    """Create basic participant analyzer"""
    config = ParticipantConfig(
        detect_interruptions=False,
        detect_questions=True,
        detect_agreements=True,
        detect_topic_initiations=False,
        include_temporal_analysis=False,
        include_interaction_matrix=False
    )
    return ParticipantAnalyzer(config)


def create_comprehensive_participant_analyzer() -> ParticipantAnalyzer:
    """Create comprehensive participant analyzer"""
    config = ParticipantConfig(
        detect_interruptions=True,
        detect_questions=True,
        detect_agreements=True,
        detect_topic_initiations=True,
        include_temporal_analysis=True,
        include_interaction_matrix=True
    )
    return ParticipantAnalyzer(config)


def create_fast_participant_analyzer() -> ParticipantAnalyzer:
    """Create fast participant analyzer"""
    config = ParticipantConfig(
        detect_interruptions=False,
        detect_questions=False,
        detect_agreements=False,
        detect_topic_initiations=False,
        include_temporal_analysis=False,
        include_interaction_matrix=False
    )
    return ParticipantAnalyzer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Participant Analyzer Test")
    parser.add_argument("--transcript", type=str, required=True, help="Meeting transcript")
    parser.add_argument("--analyzer-type", type=str, default="comprehensive",
                       choices=["basic", "comprehensive", "fast"], help="Analyzer type")
    args = parser.parse_args()
    
    # Create analyzer
    if args.analyzer_type == "basic":
        analyzer = create_basic_participant_analyzer()
    elif args.analyzer_type == "fast":
        analyzer = create_fast_participant_analyzer()
    else:
        analyzer = create_comprehensive_participant_analyzer()
        
    try:
        print(f"Analyzer status: {analyzer.get_status()}")
        
        # Analyze participants
        print("Analyzing participants...")
        result = analyzer.analyze_participants(args.transcript)
        
        if result.success:
            print(f"Analysis completed successfully!")
            print(f"Analysis ID: {result.analysis_id}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Total participants: {result.total_participants}")
            print(f"Overall engagement: {result.overall_engagement:.3f}")
            print(f"Participation balance: {result.participation_balance:.3f}")
            
            if result.participant_stats:
                print(f"\nParticipant Statistics:")
                for participant in result.participant_stats:
                    print(f"  {participant.name}:")
                    print(f"    Speaking time: {participant.speaking_time_percentage:.1f}%")
                    print(f"    Word count: {participant.word_count}")
                    print(f"    Engagement: {participant.engagement_metrics.engagement_level.value}")
                    print(f"    Pattern: {participant.participation_pattern.value}")
                    
        else:
            print(f"Analysis failed: {result.errors}")
            
    except Exception as e:
        print(f"Error: {e}")