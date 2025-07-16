#!/usr/bin/env python3
"""
Confidence Scorer Module

Quality assessment system that evaluates confidence and reliability of AI outputs
across transcription and analysis stages with comprehensive scoring metrics.

Author: Claude AI Assistant
Date: 2025-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
import re
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from statistics import mean, stdev

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoreType(Enum):
    """Types of confidence scores"""
    TRANSCRIPTION = "transcription"
    SPEAKER_DIARIZATION = "speaker_diarization"
    LLM_ANALYSIS = "llm_analysis"
    PARTICIPANT_ANALYSIS = "participant_analysis"
    OVERALL_PIPELINE = "overall_pipeline"


class QualityLevel(Enum):
    """Quality levels for assessments"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for different components"""
    
    # Component scores
    transcription_confidence: float = 0.0
    speaker_diarization_confidence: float = 0.0
    llm_analysis_confidence: float = 0.0
    participant_analysis_confidence: float = 0.0
    
    # Quality indicators
    audio_quality_score: float = 0.0
    content_coherence_score: float = 0.0
    analysis_completeness_score: float = 0.0
    
    # Consistency metrics
    internal_consistency: float = 0.0
    cross_component_consistency: float = 0.0
    
    # Reliability indicators
    data_sufficiency: float = 0.0
    processing_stability: float = 0.0
    
    # Combined scores
    overall_confidence: float = 0.0
    quality_level: QualityLevel = QualityLevel.FAIR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "transcription_confidence": self.transcription_confidence,
            "speaker_diarization_confidence": self.speaker_diarization_confidence,
            "llm_analysis_confidence": self.llm_analysis_confidence,
            "participant_analysis_confidence": self.participant_analysis_confidence,
            "audio_quality_score": self.audio_quality_score,
            "content_coherence_score": self.content_coherence_score,
            "analysis_completeness_score": self.analysis_completeness_score,
            "internal_consistency": self.internal_consistency,
            "cross_component_consistency": self.cross_component_consistency,
            "data_sufficiency": self.data_sufficiency,
            "processing_stability": self.processing_stability,
            "overall_confidence": self.overall_confidence,
            "quality_level": self.quality_level.value
        }


@dataclass
class QualityAssessment:
    """Quality assessment for a specific component"""
    
    component: str
    score: float
    quality_level: QualityLevel
    
    # Detailed metrics
    accuracy_indicators: Dict[str, float] = field(default_factory=dict)
    completeness_indicators: Dict[str, float] = field(default_factory=dict)
    consistency_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Supporting evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "component": self.component,
            "score": self.score,
            "quality_level": self.quality_level.value,
            "accuracy_indicators": self.accuracy_indicators,
            "completeness_indicators": self.completeness_indicators,
            "consistency_indicators": self.consistency_indicators,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "evidence": self.evidence
        }


@dataclass
class ScoreConfig:
    """Configuration for confidence scoring"""
    
    # Scoring weights
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "transcription": 0.35,
        "speaker_diarization": 0.15,
        "llm_analysis": 0.35,
        "participant_analysis": 0.15
    })
    
    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 0.9,
        "good": 0.75,
        "fair": 0.6,
        "poor": 0.4
    })
    
    # Scoring factors
    scoring_factors: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.4,
        "completeness": 0.3,
        "consistency": 0.2,
        "reliability": 0.1
    })
    
    # Validation settings
    min_confidence_threshold: float = 0.5
    enable_cross_validation: bool = True
    require_evidence: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "component_weights": self.component_weights,
            "quality_thresholds": self.quality_thresholds,
            "scoring_factors": self.scoring_factors,
            "min_confidence_threshold": self.min_confidence_threshold,
            "enable_cross_validation": self.enable_cross_validation,
            "require_evidence": self.require_evidence
        }


@dataclass
class ValidationResult:
    """Result from confidence validation"""
    
    # Validation metadata
    validation_id: str
    timestamp: datetime
    
    # Validation results
    is_valid: bool = True
    confidence_score: float = 0.0
    quality_assessment: QualityAssessment = None
    
    # Validation details
    validation_checks: Dict[str, bool] = field(default_factory=dict)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "validation_id": self.validation_id,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "confidence_score": self.confidence_score,
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "validation_checks": self.validation_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "improvement_suggestions": self.improvement_suggestions
        }


class ConfidenceScorer:
    """AI output confidence assessment system"""
    
    def __init__(self, config: Optional[ScoreConfig] = None):
        self.config = config or ScoreConfig()
        self.scorer_id = str(uuid.uuid4())
        
        # Scoring patterns
        self.uncertainty_patterns = [
            r'(?i)\b(maybe|perhaps|possibly|might|could|probably|likely|uncertain|unclear)\b',
            r'(?i)\b(i think|i believe|seems like|appears to|sounds like)\b',
            r'(?i)\b(not sure|don\'t know|hard to say|difficult to determine)\b'
        ]
        
        self.confidence_patterns = [
            r'(?i)\b(definitely|certainly|clearly|obviously|absolutely|exactly|precisely)\b',
            r'(?i)\b(confirmed|established|determined|concluded|decided)\b',
            r'(?i)\b(without doubt|no question|for sure|beyond doubt)\b'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'coherence': [
                r'(?i)\b(therefore|thus|consequently|as a result|in conclusion)\b',
                r'(?i)\b(first|second|third|next|then|finally)\b',
                r'(?i)\b(however|but|although|despite|nevertheless)\b'
            ],
            'completeness': [
                r'(?i)\b(in summary|to summarize|overall|in total)\b',
                r'(?i)\b(all|every|complete|comprehensive|thorough)\b'
            ],
            'specificity': [
                r'\b\d+\b',  # Numbers
                r'(?i)\b(specific|particular|exact|precise|detailed)\b',
                r'(?i)\b(at \d+|on \d+|by \d+|during \d+)\b'  # Time references
            ]
        }
        
        # Statistics
        self.stats = {
            "total_scorings": 0,
            "average_confidence": 0.0,
            "quality_distribution": {level.value: 0 for level in QualityLevel},
            "component_performance": {}
        }
        
        logger.info(f"ConfidenceScorer initialized with ID: {self.scorer_id}")
        
    def score_pipeline_result(self, pipeline_result: Any) -> ConfidenceMetrics:
        """Score confidence for complete pipeline result"""
        try:
            logger.info("Scoring pipeline result confidence")
            
            # Update statistics
            self.stats["total_scorings"] += 1
            
            # Initialize metrics
            metrics = ConfidenceMetrics()
            
            # Score individual components
            if hasattr(pipeline_result, 'transcript') and pipeline_result.transcript:
                metrics.transcription_confidence = self._score_transcription_confidence(
                    pipeline_result.transcript, 
                    getattr(pipeline_result, 'audio_duration', 0)
                )
                
            if hasattr(pipeline_result, 'speaker_diarization') and pipeline_result.speaker_diarization:
                metrics.speaker_diarization_confidence = self._score_speaker_diarization_confidence(
                    pipeline_result.speaker_diarization,
                    pipeline_result.transcript
                )
                
            if hasattr(pipeline_result, 'llm_analysis') and pipeline_result.llm_analysis:
                metrics.llm_analysis_confidence = self._score_llm_analysis_confidence(
                    pipeline_result.llm_analysis,
                    pipeline_result.transcript
                )
                
            if hasattr(pipeline_result, 'participant_analysis') and pipeline_result.participant_analysis:
                metrics.participant_analysis_confidence = self._score_participant_analysis_confidence(
                    pipeline_result.participant_analysis,
                    pipeline_result.transcript
                )
                
            # Calculate quality scores
            metrics.audio_quality_score = self._assess_audio_quality(pipeline_result)
            metrics.content_coherence_score = self._assess_content_coherence(pipeline_result.transcript)
            metrics.analysis_completeness_score = self._assess_analysis_completeness(pipeline_result)
            
            # Calculate consistency scores
            metrics.internal_consistency = self._assess_internal_consistency(pipeline_result)
            metrics.cross_component_consistency = self._assess_cross_component_consistency(pipeline_result)
            
            # Calculate reliability scores
            metrics.data_sufficiency = self._assess_data_sufficiency(pipeline_result)
            metrics.processing_stability = self._assess_processing_stability(pipeline_result)
            
            # Calculate overall confidence
            metrics.overall_confidence = self._calculate_overall_confidence(metrics)
            metrics.quality_level = self._determine_quality_level(metrics.overall_confidence)
            
            # Update statistics
            self._update_stats(metrics)
            
            logger.info(f"Pipeline confidence scoring completed: {metrics.overall_confidence:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline confidence scoring failed: {e}")
            return ConfidenceMetrics()
            
    def _score_transcription_confidence(self, transcript: str, audio_duration: float) -> float:
        """Score transcription confidence"""
        try:
            confidence_factors = []
            
            # Length consistency (words per minute)
            if audio_duration > 0:
                word_count = len(transcript.split())
                wpm = word_count / (audio_duration / 60)
                
                # Typical speaking rate is 150-200 wpm
                if 100 <= wpm <= 250:
                    confidence_factors.append(0.8)
                elif 80 <= wpm <= 300:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
                    
            # Content quality indicators
            if transcript:
                # Check for uncertainty patterns
                uncertainty_count = sum(len(re.findall(pattern, transcript)) for pattern in self.uncertainty_patterns)
                confidence_count = sum(len(re.findall(pattern, transcript)) for pattern in self.confidence_patterns)
                
                word_count = len(transcript.split())
                if word_count > 0:
                    uncertainty_ratio = uncertainty_count / word_count
                    confidence_ratio = confidence_count / word_count
                    
                    linguistic_confidence = max(0.0, 0.7 + confidence_ratio - uncertainty_ratio)
                    confidence_factors.append(linguistic_confidence)
                    
                # Check for completeness indicators
                if len(transcript) > 100:  # Minimum meaningful length
                    confidence_factors.append(0.7)
                    
                # Check for coherence
                coherence_score = self._assess_content_coherence(transcript)
                confidence_factors.append(coherence_score)
                
            return mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Transcription confidence scoring failed: {e}")
            return 0.0
            
    def _score_speaker_diarization_confidence(self, diarization_data: Dict[str, Any], transcript: str) -> float:
        """Score speaker diarization confidence"""
        try:
            confidence_factors = []
            
            # Check if diarization data exists
            if not diarization_data:
                return 0.0
                
            # Speaker count consistency
            speakers = diarization_data.get('speakers', [])
            if speakers:
                speaker_count = len(speakers)
                
                # Reasonable speaker count for meetings
                if 2 <= speaker_count <= 10:
                    confidence_factors.append(0.8)
                elif speaker_count == 1:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
                    
            # Segment consistency
            segments = diarization_data.get('segments', [])
            if segments:
                # Check for reasonable segment distribution
                segment_lengths = [len(seg.get('text', '')) for seg in segments]
                if segment_lengths:
                    avg_length = mean(segment_lengths)
                    if avg_length > 10:  # Reasonable segment length
                        confidence_factors.append(0.7)
                    else:
                        confidence_factors.append(0.4)
                        
                # Check for speaker transitions
                speaker_changes = 0
                prev_speaker = None
                for seg in segments:
                    current_speaker = seg.get('speaker')
                    if prev_speaker and current_speaker != prev_speaker:
                        speaker_changes += 1
                    prev_speaker = current_speaker
                    
                if speaker_changes > 0:
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.5)
                    
            return mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Speaker diarization confidence scoring failed: {e}")
            return 0.0
            
    def _score_llm_analysis_confidence(self, analysis_data: Dict[str, Any], transcript: str) -> float:
        """Score LLM analysis confidence"""
        try:
            confidence_factors = []
            
            # Check analysis components
            if 'analysis_result' in analysis_data:
                analysis_result = analysis_data['analysis_result']
                
                # Summary quality
                if 'summary' in analysis_result:
                    summary = analysis_result['summary']
                    if summary and len(summary) > 50:
                        confidence_factors.append(0.8)
                    else:
                        confidence_factors.append(0.4)
                        
                # Action items quality
                if 'action_items' in analysis_result:
                    action_items = analysis_result['action_items']
                    if action_items and len(action_items) > 0:
                        confidence_factors.append(0.7)
                    else:
                        confidence_factors.append(0.5)
                        
                # Topics quality
                if 'topics' in analysis_result:
                    topics = analysis_result['topics']
                    if topics and len(topics) > 0:
                        confidence_factors.append(0.6)
                    else:
                        confidence_factors.append(0.4)
                        
            # Check for LLM-specific confidence indicators
            if 'confidence_scores' in analysis_data:
                llm_confidence = analysis_data['confidence_scores']
                if isinstance(llm_confidence, dict):
                    scores = [v for v in llm_confidence.values() if isinstance(v, (int, float))]
                    if scores:
                        confidence_factors.append(mean(scores))
                        
            # Analysis completeness
            completeness_score = self._assess_analysis_completeness(analysis_data)
            confidence_factors.append(completeness_score)
            
            return mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"LLM analysis confidence scoring failed: {e}")
            return 0.0
            
    def _score_participant_analysis_confidence(self, participant_data: Dict[str, Any], transcript: str) -> float:
        """Score participant analysis confidence"""
        try:
            confidence_factors = []
            
            # Check if speaker labels exist
            if not (":" in transcript):
                return 0.2  # Low confidence without speaker labels
                
            # Participant count
            if 'participant_stats' in participant_data:
                participant_stats = participant_data['participant_stats']
                if participant_stats and len(participant_stats) > 0:
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.3)
                    
            # Data richness
            if 'interaction_matrix' in participant_data:
                interaction_matrix = participant_data['interaction_matrix']
                if interaction_matrix:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)
                    
            # Analysis quality indicators
            if 'overall_engagement' in participant_data:
                engagement = participant_data['overall_engagement']
                if isinstance(engagement, (int, float)) and engagement > 0:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
                    
            return mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Participant analysis confidence scoring failed: {e}")
            return 0.0
            
    def _assess_audio_quality(self, pipeline_result: Any) -> float:
        """Assess audio quality indicators"""
        try:
            quality_factors = []
            
            # Audio duration
            if hasattr(pipeline_result, 'audio_duration') and pipeline_result.audio_duration > 0:
                duration = pipeline_result.audio_duration
                if 30 <= duration <= 3600:  # 30 seconds to 1 hour
                    quality_factors.append(0.8)
                elif duration > 3600:  # Very long meetings
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.4)
                    
            # Transcript length vs duration consistency
            if hasattr(pipeline_result, 'transcript') and hasattr(pipeline_result, 'audio_duration'):
                if pipeline_result.transcript and pipeline_result.audio_duration > 0:
                    word_count = len(pipeline_result.transcript.split())
                    expected_words = pipeline_result.audio_duration / 60 * 150  # 150 wpm average
                    
                    ratio = word_count / expected_words if expected_words > 0 else 0
                    if 0.5 <= ratio <= 2.0:  # Reasonable range
                        quality_factors.append(0.7)
                    else:
                        quality_factors.append(0.4)
                        
            return mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}")
            return 0.5
            
    def _assess_content_coherence(self, transcript: str) -> float:
        """Assess content coherence"""
        try:
            if not transcript:
                return 0.0
                
            coherence_factors = []
            
            # Check for coherence indicators
            for indicator_type, patterns in self.quality_indicators.items():
                count = sum(len(re.findall(pattern, transcript)) for pattern in patterns)
                word_count = len(transcript.split())
                
                if word_count > 0:
                    indicator_density = count / word_count
                    
                    if indicator_type == 'coherence':
                        coherence_factors.append(min(1.0, indicator_density * 50))
                    elif indicator_type == 'specificity':
                        coherence_factors.append(min(1.0, indicator_density * 20))
                        
            # Check for repeated phrases (may indicate transcription issues)
            sentences = transcript.split('.')
            if len(sentences) > 1:
                unique_sentences = set(sentences)
                uniqueness_ratio = len(unique_sentences) / len(sentences)
                coherence_factors.append(uniqueness_ratio)
                
            return mean(coherence_factors) if coherence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Content coherence assessment failed: {e}")
            return 0.5
            
    def _assess_analysis_completeness(self, pipeline_result: Any) -> float:
        """Assess analysis completeness"""
        try:
            completeness_factors = []
            
            # Check for expected components
            expected_components = ['transcript', 'llm_analysis', 'speaker_diarization']
            present_components = 0
            
            for component in expected_components:
                if hasattr(pipeline_result, component) and getattr(pipeline_result, component):
                    present_components += 1
                    
            if expected_components:
                completeness_factors.append(present_components / len(expected_components))
                
            # Check LLM analysis completeness
            if hasattr(pipeline_result, 'llm_analysis') and pipeline_result.llm_analysis:
                analysis = pipeline_result.llm_analysis
                if isinstance(analysis, dict):
                    analysis_components = ['summary', 'action_items', 'topics']
                    present_analysis = sum(1 for comp in analysis_components if comp in analysis and analysis[comp])
                    completeness_factors.append(present_analysis / len(analysis_components))
                    
            return mean(completeness_factors) if completeness_factors else 0.0
            
        except Exception as e:
            logger.error(f"Analysis completeness assessment failed: {e}")
            return 0.0
            
    def _assess_internal_consistency(self, pipeline_result: Any) -> float:
        """Assess internal consistency"""
        try:
            consistency_factors = []
            
            # Check transcript vs analysis consistency
            if hasattr(pipeline_result, 'transcript') and hasattr(pipeline_result, 'llm_analysis'):
                transcript = pipeline_result.transcript
                analysis = pipeline_result.llm_analysis
                
                if transcript and analysis:
                    # Check if summary relates to transcript
                    if isinstance(analysis, dict) and 'summary' in analysis:
                        summary = analysis['summary']
                        if isinstance(summary, str) and len(summary) > 10:
                            # Simple consistency check - shared words
                            transcript_words = set(transcript.lower().split())
                            summary_words = set(summary.lower().split())
                            
                            if transcript_words and summary_words:
                                overlap = len(transcript_words & summary_words)
                                consistency_ratio = overlap / min(len(transcript_words), len(summary_words))
                                consistency_factors.append(min(1.0, consistency_ratio * 2))
                                
            # Check speaker diarization consistency
            if hasattr(pipeline_result, 'speaker_diarization') and hasattr(pipeline_result, 'participant_analysis'):
                diarization = pipeline_result.speaker_diarization
                participant_analysis = pipeline_result.participant_analysis
                
                if diarization and participant_analysis:
                    # Check if speaker counts match
                    diarization_speakers = len(diarization.get('speakers', []))
                    participant_speakers = len(participant_analysis.get('participant_stats', []))
                    
                    if diarization_speakers > 0 and participant_speakers > 0:
                        speaker_consistency = min(diarization_speakers, participant_speakers) / max(diarization_speakers, participant_speakers)
                        consistency_factors.append(speaker_consistency)
                        
            return mean(consistency_factors) if consistency_factors else 0.5
            
        except Exception as e:
            logger.error(f"Internal consistency assessment failed: {e}")
            return 0.5
            
    def _assess_cross_component_consistency(self, pipeline_result: Any) -> float:
        """Assess cross-component consistency"""
        try:
            consistency_factors = []
            
            # Check processing times consistency
            if hasattr(pipeline_result, 'processing_time') and hasattr(pipeline_result, 'audio_duration'):
                processing_time = pipeline_result.processing_time
                audio_duration = pipeline_result.audio_duration
                
                if processing_time > 0 and audio_duration > 0:
                    # Processing should be reasonable compared to audio duration
                    time_ratio = processing_time / audio_duration
                    # Expect processing to be 1-5x audio duration
                    if 1.0 <= time_ratio <= 5.0:
                        consistency_factors.append(1.0)
                    elif 0.5 <= time_ratio <= 10.0:
                        consistency_factors.append(0.7)
                    else:
                        consistency_factors.append(0.3)
                        
            # Check confidence scores consistency
            if hasattr(pipeline_result, 'confidence_scores'):
                scores = pipeline_result.confidence_scores
                if isinstance(scores, dict) and scores:
                    score_values = [v for v in scores.values() if isinstance(v, (int, float))]
                    if score_values:
                        # Check if scores are within reasonable range
                        score_variance = stdev(score_values) if len(score_values) > 1 else 0
                        if score_variance < 0.3:
                            consistency_factors.append(0.9)
                        elif score_variance < 0.5:
                            consistency_factors.append(0.7)
                        else:
                            consistency_factors.append(0.5)
                            
            return mean(consistency_factors) if consistency_factors else 0.5
            
        except Exception as e:
            logger.error(f"Cross-component consistency assessment failed: {e}")
            return 0.5
            
    def _assess_data_sufficiency(self, pipeline_result: Any) -> float:
        """Assess data sufficiency"""
        try:
            sufficiency_factors = []
            
            # Check transcript length
            if hasattr(pipeline_result, 'transcript'):
                transcript = pipeline_result.transcript
                if isinstance(transcript, str):
                    word_count = len(transcript.split())
                    if word_count >= 100:
                        sufficiency_factors.append(1.0)
                    elif word_count >= 50:
                        sufficiency_factors.append(0.7)
                    elif word_count >= 20:
                        sufficiency_factors.append(0.5)
                    else:
                        sufficiency_factors.append(0.2)
                        
            # Check audio duration
            if hasattr(pipeline_result, 'audio_duration'):
                duration = pipeline_result.audio_duration
                if isinstance(duration, (int, float)):
                    if duration >= 300:  # 5 minutes
                        sufficiency_factors.append(1.0)
                    elif duration >= 120:  # 2 minutes
                        sufficiency_factors.append(0.8)
                    elif duration >= 60:  # 1 minute
                        sufficiency_factors.append(0.6)
                    else:
                        sufficiency_factors.append(0.3)
                        
            # Check speaker diversity
            if hasattr(pipeline_result, 'speaker_diarization'):
                diarization = pipeline_result.speaker_diarization
                if isinstance(diarization, dict):
                    speaker_count = len(diarization.get('speakers', []))
                    if speaker_count >= 3:
                        sufficiency_factors.append(1.0)
                    elif speaker_count >= 2:
                        sufficiency_factors.append(0.8)
                    elif speaker_count >= 1:
                        sufficiency_factors.append(0.6)
                    else:
                        sufficiency_factors.append(0.3)
                        
            return mean(sufficiency_factors) if sufficiency_factors else 0.5
            
        except Exception as e:
            logger.error(f"Data sufficiency assessment failed: {e}")
            return 0.5
            
    def _assess_processing_stability(self, pipeline_result: Any) -> float:
        """Assess processing stability"""
        try:
            stability_factors = []
            
            # Check for errors
            if hasattr(pipeline_result, 'errors'):
                errors = pipeline_result.errors
                if isinstance(errors, list):
                    if len(errors) == 0:
                        stability_factors.append(1.0)
                    elif len(errors) <= 2:
                        stability_factors.append(0.7)
                    elif len(errors) <= 5:
                        stability_factors.append(0.5)
                    else:
                        stability_factors.append(0.2)
                        
            # Check for warnings
            if hasattr(pipeline_result, 'warnings'):
                warnings = pipeline_result.warnings
                if isinstance(warnings, list):
                    if len(warnings) == 0:
                        stability_factors.append(1.0)
                    elif len(warnings) <= 3:
                        stability_factors.append(0.8)
                    elif len(warnings) <= 7:
                        stability_factors.append(0.6)
                    else:
                        stability_factors.append(0.4)
                        
            # Check processing status
            if hasattr(pipeline_result, 'status'):
                status = pipeline_result.status
                if isinstance(status, str):
                    if status.lower() in ['completed', 'success']:
                        stability_factors.append(1.0)
                    elif status.lower() in ['partial', 'warning']:
                        stability_factors.append(0.7)
                    elif status.lower() in ['failed', 'error']:
                        stability_factors.append(0.2)
                    else:
                        stability_factors.append(0.5)
                        
            return mean(stability_factors) if stability_factors else 0.5
            
        except Exception as e:
            logger.error(f"Processing stability assessment failed: {e}")
            return 0.5
            
    def _calculate_overall_confidence(self, metrics: ConfidenceMetrics) -> float:
        """Calculate overall confidence score"""
        try:
            # Component confidence scores
            component_scores = [
                metrics.transcription_confidence * self.config.component_weights["transcription"],
                metrics.speaker_diarization_confidence * self.config.component_weights["speaker_diarization"],
                metrics.llm_analysis_confidence * self.config.component_weights["llm_analysis"],
                metrics.participant_analysis_confidence * self.config.component_weights["participant_analysis"]
            ]
            
            weighted_component_score = sum(component_scores)
            
            # Quality scores
            quality_scores = [
                metrics.audio_quality_score,
                metrics.content_coherence_score,
                metrics.analysis_completeness_score
            ]
            
            average_quality_score = mean(quality_scores)
            
            # Reliability scores
            reliability_scores = [
                metrics.internal_consistency,
                metrics.cross_component_consistency,
                metrics.data_sufficiency,
                metrics.processing_stability
            ]
            
            average_reliability_score = mean(reliability_scores)
            
            # Combine all scores
            overall_confidence = (
                weighted_component_score * 0.6 +
                average_quality_score * 0.25 +
                average_reliability_score * 0.15
            )
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Overall confidence calculation failed: {e}")
            return 0.0
            
    def _determine_quality_level(self, confidence_score: float) -> QualityLevel:
        """Determine quality level from confidence score"""
        try:
            thresholds = self.config.quality_thresholds
            
            if confidence_score >= thresholds["excellent"]:
                return QualityLevel.EXCELLENT
            elif confidence_score >= thresholds["good"]:
                return QualityLevel.GOOD
            elif confidence_score >= thresholds["fair"]:
                return QualityLevel.FAIR
            elif confidence_score >= thresholds["poor"]:
                return QualityLevel.POOR
            else:
                return QualityLevel.FAILED
                
        except Exception as e:
            logger.error(f"Quality level determination failed: {e}")
            return QualityLevel.FAIR
            
    def _update_stats(self, metrics: ConfidenceMetrics):
        """Update scorer statistics"""
        try:
            # Update average confidence
            total = self.stats["total_scorings"]
            old_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (
                old_avg * (total - 1) + metrics.overall_confidence
            ) / total
            
            # Update quality distribution
            self.stats["quality_distribution"][metrics.quality_level.value] += 1
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            
    def assess_component_quality(self, component: str, data: Any) -> QualityAssessment:
        """Assess quality of a specific component"""
        try:
            assessment = QualityAssessment(component=component, score=0.0, quality_level=QualityLevel.FAIR)
            
            # Component-specific assessment
            if component == "transcription":
                assessment.score = self._score_transcription_confidence(data, 0)
            elif component == "speaker_diarization":
                assessment.score = self._score_speaker_diarization_confidence(data, "")
            elif component == "llm_analysis":
                assessment.score = self._score_llm_analysis_confidence(data, "")
            elif component == "participant_analysis":
                assessment.score = self._score_participant_analysis_confidence(data, "")
                
            assessment.quality_level = self._determine_quality_level(assessment.score)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Component quality assessment failed: {e}")
            return QualityAssessment(component=component, score=0.0, quality_level=QualityLevel.FAILED)
            
    def validate_confidence(self, confidence_score: float, evidence: Dict[str, Any]) -> ValidationResult:
        """Validate confidence score with evidence"""
        validation_id = str(uuid.uuid4())
        
        result = ValidationResult(
            validation_id=validation_id,
            timestamp=datetime.now(),
            confidence_score=confidence_score
        )
        
        try:
            # Validation checks
            checks = {}
            
            # Minimum threshold check
            checks["min_threshold"] = confidence_score >= self.config.min_confidence_threshold
            
            # Evidence requirement check
            if self.config.require_evidence:
                checks["has_evidence"] = bool(evidence)
            else:
                checks["has_evidence"] = True
                
            # Consistency check
            checks["internal_consistency"] = True  # Placeholder
            
            result.validation_checks = checks
            result.failed_checks = [check for check, passed in checks.items() if not passed]
            result.is_valid = len(result.failed_checks) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"Confidence validation failed: {e}")
            result.is_valid = False
            result.failed_checks = ["validation_error"]
            return result
            
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get scorer status"""
        return {
            "scorer_id": self.scorer_id,
            "config": self.config.to_dict(),
            "stats": self.get_stats()
        }


# Factory functions  
def create_basic_confidence_scorer() -> ConfidenceScorer:
    """Create basic confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.4,
        enable_cross_validation=False,
        require_evidence=False
    )
    return ConfidenceScorer(config)


def create_strict_confidence_scorer() -> ConfidenceScorer:
    """Create strict confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.7,
        enable_cross_validation=True,
        require_evidence=True,
        quality_thresholds={
            "excellent": 0.95,
            "good": 0.85,
            "fair": 0.7,
            "poor": 0.5
        }
    )
    return ConfidenceScorer(config)


def create_balanced_confidence_scorer() -> ConfidenceScorer:
    """Create balanced confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.6,
        enable_cross_validation=True,
        require_evidence=True
    )
    return ConfidenceScorer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Confidence Scorer Test")
    parser.add_argument("--scorer-type", type=str, default="balanced",
                       choices=["basic", "strict", "balanced"], help="Scorer type")
    args = parser.parse_args()
    
    # Create scorer
    if args.scorer_type == "basic":
        scorer = create_basic_confidence_scorer()
    elif args.scorer_type == "strict":
        scorer = create_strict_confidence_scorer()
    else:
        scorer = create_balanced_confidence_scorer()
        
    try:
        print(f"Scorer status: {scorer.get_status()}")
        
        # Create mock pipeline result for testing
        class MockPipelineResult:
            def __init__(self):
                self.transcript = "This is a test meeting transcript with multiple speakers discussing various topics."
                self.audio_duration = 120.0
                self.speaker_diarization = {
                    'speakers': ['Speaker 1', 'Speaker 2'],
                    'segments': [
                        {'speaker': 'Speaker 1', 'text': 'Hello everyone'},
                        {'speaker': 'Speaker 2', 'text': 'Thank you for joining'}
                    ]
                }
                self.llm_analysis = {
                    'analysis_result': {
                        'summary': 'This meeting discussed project planning and next steps.',
                        'action_items': ['Review project timeline', 'Prepare budget proposal'],
                        'topics': ['Project Planning', 'Budget Discussion']
                    }
                }
                self.participant_analysis = {
                    'participant_stats': [
                        {'name': 'Speaker 1', 'speaking_time_percentage': 60.0},
                        {'name': 'Speaker 2', 'speaking_time_percentage': 40.0}
                    ],
                    'overall_engagement': 0.75
                }
                self.confidence_scores = {'overall': 0.8}
                self.processing_time = 45.0
                self.errors = []
                self.warnings = []
                self.status = 'completed'
                
        # Test confidence scoring
        mock_result = MockPipelineResult()
        print("Scoring pipeline result...")
        
        metrics = scorer.score_pipeline_result(mock_result)
        
        print(f"Confidence scoring completed!")
        print(f"Overall confidence: {metrics.overall_confidence:.3f}")
        print(f"Quality level: {metrics.quality_level.value}")
        print(f"Transcription confidence: {metrics.transcription_confidence:.3f}")
        print(f"LLM analysis confidence: {metrics.llm_analysis_confidence:.3f}")
        print(f"Content coherence: {metrics.content_coherence_score:.3f}")
        print(f"Analysis completeness: {metrics.analysis_completeness_score:.3f}")
        
        # Test validation
        print("\nValidating confidence...")
        validation = scorer.validate_confidence(metrics.overall_confidence, {"test": "evidence"})
        print(f"Validation result: {validation.is_valid}")
        if not validation.is_valid:
            print(f"Failed checks: {validation.failed_checks}")
            
    except Exception as e:
        print(f"Error: {e}")
                        consistency_factors.append(0.4)
                        
            # Check confidence scores consistency
            if hasattr(pipeline_result, 'confidence_scores'):
                confidence_scores = pipeline_result.confidence_scores
                if isinstance(confidence_scores, dict):
                    scores = [v for v in confidence_scores.values() if isinstance(v, (int, float))]
                    if len(scores) > 1:
                        score_std = stdev(scores)
                        # Lower standard deviation indicates more consistent scores
                        consistency_factors.append(max(0.0, 1.0 - score_std))
                        
            return mean(consistency_factors) if consistency_factors else 0.5
            
        except Exception as e:
            logger.error(f"Cross-component consistency assessment failed: {e}")
            return 0.5
            
    def _assess_data_sufficiency(self, pipeline_result: Any) -> float:
        """Assess data sufficiency"""
        try:
            sufficiency_factors = []
            
            # Transcript length sufficiency
            if hasattr(pipeline_result, 'transcript') and pipeline_result.transcript:
                word_count = len(pipeline_result.transcript.split())
                if word_count >= 100:
                    sufficiency_factors.append(0.8)
                elif word_count >= 50:
                    sufficiency_factors.append(0.6)
                else:
                    sufficiency_factors.append(0.3)
                    
            # Audio duration sufficiency
            if hasattr(pipeline_result, 'audio_duration') and pipeline_result.audio_duration > 0:
                duration = pipeline_result.audio_duration
                if duration >= 60:  # At least 1 minute
                    sufficiency_factors.append(0.8)
                elif duration >= 30:  # At least 30 seconds
                    sufficiency_factors.append(0.6)
                else:
                    sufficiency_factors.append(0.3)
                    
            # Speaker count sufficiency
            if hasattr(pipeline_result, 'speaker_count') and pipeline_result.speaker_count > 0:
                speaker_count = pipeline_result.speaker_count
                if speaker_count >= 2:
                    sufficiency_factors.append(0.8)
                else:
                    sufficiency_factors.append(0.5)
                    
            return mean(sufficiency_factors) if sufficiency_factors else 0.0
            
        except Exception as e:
            logger.error(f"Data sufficiency assessment failed: {e}")
            return 0.0
            
    def _assess_processing_stability(self, pipeline_result: Any) -> float:
        """Assess processing stability"""
        try:
            stability_factors = []
            
            # Check for errors
            if hasattr(pipeline_result, 'errors') and pipeline_result.errors:
                error_count = len(pipeline_result.errors)
                if error_count == 0:
                    stability_factors.append(1.0)
                elif error_count <= 2:
                    stability_factors.append(0.7)
                else:
                    stability_factors.append(0.3)
            else:
                stability_factors.append(0.8)  # Assume stable if no error info
                
            # Check for warnings
            if hasattr(pipeline_result, 'warnings') and pipeline_result.warnings:
                warning_count = len(pipeline_result.warnings)
                if warning_count == 0:
                    stability_factors.append(1.0)
                elif warning_count <= 3:
                    stability_factors.append(0.8)
                else:
                    stability_factors.append(0.5)
            else:
                stability_factors.append(0.8)  # Assume stable if no warning info
                
            # Check processing completion
            if hasattr(pipeline_result, 'status'):
                status = pipeline_result.status
                if hasattr(status, 'value'):
                    status_value = status.value
                else:
                    status_value = str(status)
                    
                if status_value == 'completed':
                    stability_factors.append(1.0)
                elif status_value == 'failed':
                    stability_factors.append(0.0)
                else:
                    stability_factors.append(0.5)
                    
            return mean(stability_factors) if stability_factors else 0.5
            
        except Exception as e:
            logger.error(f"Processing stability assessment failed: {e}")
            return 0.5
            
    def _calculate_overall_confidence(self, metrics: ConfidenceMetrics) -> float:
        """Calculate overall confidence score"""
        try:
            # Component confidence scores
            component_scores = [
                metrics.transcription_confidence * self.config.component_weights["transcription"],
                metrics.speaker_diarization_confidence * self.config.component_weights["speaker_diarization"],
                metrics.llm_analysis_confidence * self.config.component_weights["llm_analysis"],
                metrics.participant_analysis_confidence * self.config.component_weights["participant_analysis"]
            ]
            
            weighted_component_score = sum(component_scores)
            
            # Quality scores
            quality_scores = [
                metrics.audio_quality_score,
                metrics.content_coherence_score,
                metrics.analysis_completeness_score
            ]
            
            average_quality_score = mean(quality_scores)
            
            # Reliability scores
            reliability_scores = [
                metrics.internal_consistency,
                metrics.cross_component_consistency,
                metrics.data_sufficiency,
                metrics.processing_stability
            ]
            
            average_reliability_score = mean(reliability_scores)
            
            # Combine all scores
            overall_confidence = (
                weighted_component_score * 0.6 +
                average_quality_score * 0.25 +
                average_reliability_score * 0.15
            )
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Overall confidence calculation failed: {e}")
            return 0.0
            
    def _determine_quality_level(self, confidence_score: float) -> QualityLevel:
        """Determine quality level from confidence score"""
        try:
            thresholds = self.config.quality_thresholds
            
            if confidence_score >= thresholds["excellent"]:
                return QualityLevel.EXCELLENT
            elif confidence_score >= thresholds["good"]:
                return QualityLevel.GOOD
            elif confidence_score >= thresholds["fair"]:
                return QualityLevel.FAIR
            elif confidence_score >= thresholds["poor"]:
                return QualityLevel.POOR
            else:
                return QualityLevel.FAILED
                
        except Exception as e:
            logger.error(f"Quality level determination failed: {e}")
            return QualityLevel.FAIR
            
    def _update_stats(self, metrics: ConfidenceMetrics):
        """Update scorer statistics"""
        try:
            # Update average confidence
            total = self.stats["total_scorings"]
            old_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (
                old_avg * (total - 1) + metrics.overall_confidence
            ) / total
            
            # Update quality distribution
            self.stats["quality_distribution"][metrics.quality_level.value] += 1
            
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            
    def assess_component_quality(self, component: str, data: Any) -> QualityAssessment:
        """Assess quality of a specific component"""
        try:
            assessment = QualityAssessment(component=component, score=0.0, quality_level=QualityLevel.FAIR)
            
            # Component-specific assessment
            if component == "transcription":
                assessment.score = self._score_transcription_confidence(data, 0)
            elif component == "speaker_diarization":
                assessment.score = self._score_speaker_diarization_confidence(data, "")
            elif component == "llm_analysis":
                assessment.score = self._score_llm_analysis_confidence(data, "")
            elif component == "participant_analysis":
                assessment.score = self._score_participant_analysis_confidence(data, "")
                
            assessment.quality_level = self._determine_quality_level(assessment.score)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Component quality assessment failed: {e}")
            return QualityAssessment(component=component, score=0.0, quality_level=QualityLevel.FAILED)
            
    def validate_confidence(self, confidence_score: float, evidence: Dict[str, Any]) -> ValidationResult:
        """Validate confidence score with evidence"""
        validation_id = str(uuid.uuid4())
        
        result = ValidationResult(
            validation_id=validation_id,
            timestamp=datetime.now(),
            confidence_score=confidence_score
        )
        
        try:
            # Validation checks
            checks = {}
            
            # Minimum threshold check
            checks["min_threshold"] = confidence_score >= self.config.min_confidence_threshold
            
            # Evidence requirement check
            if self.config.require_evidence:
                checks["has_evidence"] = bool(evidence)
            else:
                checks["has_evidence"] = True
                
            # Consistency check
            checks["internal_consistency"] = True  # Placeholder
            
            result.validation_checks = checks
            result.failed_checks = [check for check, passed in checks.items() if not passed]
            result.is_valid = len(result.failed_checks) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"Confidence validation failed: {e}")
            result.is_valid = False
            result.failed_checks = ["validation_error"]
            return result
            
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics"""
        return self.stats.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get scorer status"""
        return {
            "scorer_id": self.scorer_id,
            "config": self.config.to_dict(),
            "stats": self.get_stats()
        }


# Factory functions
def create_basic_confidence_scorer() -> ConfidenceScorer:
    """Create basic confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.4,
        enable_cross_validation=False,
        require_evidence=False
    )
    return ConfidenceScorer(config)


def create_strict_confidence_scorer() -> ConfidenceScorer:
    """Create strict confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.7,
        enable_cross_validation=True,
        require_evidence=True,
        quality_thresholds={
            "excellent": 0.95,
            "good": 0.85,
            "fair": 0.7,
            "poor": 0.5
        }
    )
    return ConfidenceScorer(config)


def create_balanced_confidence_scorer() -> ConfidenceScorer:
    """Create balanced confidence scorer"""
    config = ScoreConfig(
        min_confidence_threshold=0.6,
        enable_cross_validation=True,
        require_evidence=True
    )
    return ConfidenceScorer(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Confidence Scorer Test")
    parser.add_argument("--scorer-type", type=str, default="balanced",
                       choices=["basic", "strict", "balanced"], help="Scorer type")
    args = parser.parse_args()
    
    # Create scorer
    if args.scorer_type == "basic":
        scorer = create_basic_confidence_scorer()
    elif args.scorer_type == "strict":
        scorer = create_strict_confidence_scorer()
    else:
        scorer = create_balanced_confidence_scorer()
        
    try:
        print(f"Scorer status: {scorer.get_status()}")
        
        # Create mock pipeline result for testing
        class MockPipelineResult:
            def __init__(self):
                self.transcript = "This is a test meeting transcript with multiple speakers discussing various topics."
                self.audio_duration = 120.0
                self.speaker_diarization = {
                    'speakers': ['Speaker 1', 'Speaker 2'],
                    'segments': [
                        {'speaker': 'Speaker 1', 'text': 'Hello everyone'},
                        {'speaker': 'Speaker 2', 'text': 'Thank you for joining'}
                    ]
                }
                self.llm_analysis = {
                    'analysis_result': {
                        'summary': 'This meeting discussed project planning and next steps.',
                        'action_items': ['Review project timeline', 'Prepare budget proposal'],
                        'topics': ['Project Planning', 'Budget Discussion']
                    }
                }
                self.participant_analysis = {
                    'participant_stats': [
                        {'name': 'Speaker 1', 'speaking_time_percentage': 60.0},
                        {'name': 'Speaker 2', 'speaking_time_percentage': 40.0}
                    ],
                    'overall_engagement': 0.75
                }
                self.confidence_scores = {'overall': 0.8}
                self.processing_time = 45.0
                self.errors = []
                self.warnings = []
                self.status = 'completed'
                
        # Test confidence scoring
        mock_result = MockPipelineResult()
        print("Scoring pipeline result...")
        
        metrics = scorer.score_pipeline_result(mock_result)
        
        print(f"Confidence scoring completed!")
        print(f"Overall confidence: {metrics.overall_confidence:.3f}")
        print(f"Quality level: {metrics.quality_level.value}")
        print(f"Transcription confidence: {metrics.transcription_confidence:.3f}")
        print(f"LLM analysis confidence: {metrics.llm_analysis_confidence:.3f}")
        print(f"Content coherence: {metrics.content_coherence_score:.3f}")
        print(f"Analysis completeness: {metrics.analysis_completeness_score:.3f}")
        
        # Test validation
        print("\nValidating confidence...")
        validation = scorer.validate_confidence(metrics.overall_confidence, {"test": "evidence"})
        print(f"Validation result: {validation.is_valid}")
        if not validation.is_valid:
            print(f"Failed checks: {validation.failed_checks}")
            
    except Exception as e:
        print(f"Error: {e}")