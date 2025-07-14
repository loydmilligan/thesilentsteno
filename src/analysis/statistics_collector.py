#!/usr/bin/env python3
"""
Audio Statistics Collection for The Silent Steno

This module provides real-time audio statistics including speaking time and
participation metrics for meeting analysis and insights.

Author: The Silent Steno Team
License: MIT
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json

# Configure logging
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of audio metrics to collect"""
    SPEAKING_TIME = "speaking_time"
    SILENCE_TIME = "silence_time"
    INTERRUPTIONS = "interruptions"
    OVERLAPS = "overlaps"
    TURN_TAKING = "turn_taking"
    VOLUME_LEVELS = "volume_levels"
    SPEECH_RATE = "speech_rate"
    PARTICIPATION = "participation"

class IntervalType(Enum):
    """Time intervals for statistics aggregation"""
    REAL_TIME = "real_time"      # Continuous updates
    MINUTE = "minute"            # Per minute aggregation
    FIVE_MINUTE = "five_minute"  # 5-minute intervals
    HOUR = "hour"               # Per hour aggregation
    SESSION = "session"         # Entire session

@dataclass
class StatisticsConfig:
    """Configuration for Statistics Collector"""
    sample_rate: int = 16000
    update_interval_ms: int = 1000  # Statistics update frequency
    rolling_window_minutes: int = 5  # Rolling window for recent statistics
    speaker_timeout_ms: int = 2000   # Time before considering speaker change
    overlap_threshold_ms: int = 500  # Minimum overlap to count as interruption
    silence_threshold_ms: int = 1000 # Minimum silence to count
    volume_smoothing_factor: float = 0.1  # Exponential smoothing for volume
    enable_real_time_updates: bool = True
    store_detailed_history: bool = True
    max_history_items: int = 10000  # Maximum history items to keep
    
    def __post_init__(self):
        """Validate configuration parameters"""
        self.update_interval_s = self.update_interval_ms / 1000.0
        self.rolling_window_s = self.rolling_window_minutes * 60.0
        self.speaker_timeout_s = self.speaker_timeout_ms / 1000.0

@dataclass 
class SpeakingTimeStats:
    """Statistics for speaking time analysis"""
    speaker_id: str
    total_speaking_time_ms: float
    total_silence_time_ms: float
    speaking_percentage: float
    turn_count: int
    average_turn_duration_ms: float
    longest_turn_ms: float
    shortest_turn_ms: float
    interruptions_made: int
    interruptions_received: int
    overlaps_participated: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'speaker_id': self.speaker_id,
            'total_speaking_time_ms': self.total_speaking_time_ms,
            'total_silence_time_ms': self.total_silence_time_ms,
            'speaking_percentage': self.speaking_percentage,
            'turn_count': self.turn_count,
            'average_turn_duration_ms': self.average_turn_duration_ms,
            'longest_turn_ms': self.longest_turn_ms,
            'shortest_turn_ms': self.shortest_turn_ms,
            'interruptions_made': self.interruptions_made,
            'interruptions_received': self.interruptions_received,
            'overlaps_participated': self.overlaps_participated
        }

@dataclass
class ParticipationMetrics:
    """Metrics for meeting participation analysis"""
    total_speakers: int
    active_speakers: int  # Speakers with significant contribution
    total_session_time_ms: float
    total_speaking_time_ms: float
    total_silence_time_ms: float
    speaking_ratio: float  # Speaking time / total time
    silence_ratio: float   # Silence time / total time
    dominant_speaker: Optional[str]
    most_balanced_period_start_ms: Optional[float]
    most_balanced_period_duration_ms: Optional[float]
    turn_taking_frequency: float  # Turns per minute
    average_turn_gap_ms: float    # Average gap between speakers
    interruption_rate: float      # Interruptions per minute
    participation_balance_score: float  # 0-1, higher = more balanced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_speakers': self.total_speakers,
            'active_speakers': self.active_speakers,
            'total_session_time_ms': self.total_session_time_ms,
            'total_speaking_time_ms': self.total_speaking_time_ms,
            'total_silence_time_ms': self.total_silence_time_ms,
            'speaking_ratio': self.speaking_ratio,
            'silence_ratio': self.silence_ratio,
            'dominant_speaker': self.dominant_speaker,
            'most_balanced_period_start_ms': self.most_balanced_period_start_ms,
            'most_balanced_period_duration_ms': self.most_balanced_period_duration_ms,
            'turn_taking_frequency': self.turn_taking_frequency,
            'average_turn_gap_ms': self.average_turn_gap_ms,
            'interruption_rate': self.interruption_rate,
            'participation_balance_score': self.participation_balance_score
        }

@dataclass
class AudioStatistics:
    """Comprehensive audio statistics"""
    timestamp: float
    interval_type: IntervalType
    interval_duration_ms: float
    
    # Speaking time statistics by speaker
    speaking_stats: Dict[str, SpeakingTimeStats]
    
    # Overall participation metrics
    participation_metrics: ParticipationMetrics
    
    # Volume and quality metrics
    average_volume_db: float
    peak_volume_db: float
    volume_variance: float
    signal_to_noise_ratio: float
    
    # Temporal patterns
    speech_rate_wpm: float  # Words per minute estimate
    pause_frequency: float   # Pauses per minute
    average_pause_duration_ms: float
    
    # Meeting dynamics
    concurrent_speakers: int
    speaker_changes: int
    total_interruptions: int
    total_overlaps: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'interval_type': self.interval_type.value,
            'interval_duration_ms': self.interval_duration_ms,
            'speaking_stats': {k: v.to_dict() for k, v in self.speaking_stats.items()},
            'participation_metrics': self.participation_metrics.to_dict(),
            'average_volume_db': self.average_volume_db,
            'peak_volume_db': self.peak_volume_db,
            'volume_variance': self.volume_variance,
            'signal_to_noise_ratio': self.signal_to_noise_ratio,
            'speech_rate_wpm': self.speech_rate_wpm,
            'pause_frequency': self.pause_frequency,
            'average_pause_duration_ms': self.average_pause_duration_ms,
            'concurrent_speakers': self.concurrent_speakers,
            'speaker_changes': self.speaker_changes,
            'total_interruptions': self.total_interruptions,
            'total_overlaps': self.total_overlaps
        }

@dataclass
class SpeakerEvent:
    """Event in speaker timeline"""
    timestamp: float
    speaker_id: Optional[str]
    event_type: str  # 'start', 'end', 'change', 'overlap_start', 'overlap_end'
    volume_db: float
    confidence: float
    duration_ms: Optional[float] = None

class StatisticsCollector:
    """
    Real-time audio statistics collection system
    
    This collector analyzes audio streams to provide comprehensive statistics
    about speaking patterns, participation, and meeting dynamics.
    """
    
    def __init__(self, config: Optional[StatisticsConfig] = None):
        """
        Initialize Statistics Collector
        
        Args:
            config: Statistics collection configuration
        """
        self.config = config or StatisticsConfig()
        
        # State tracking
        self.is_running = False
        self.session_start_time = None
        self.last_update_time = None
        
        # Speaker tracking
        self.current_speakers: Dict[str, float] = {}  # speaker_id -> start_time
        self.speaker_events: List[SpeakerEvent] = []
        self.speaker_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_speaking_time_ms': 0.0,
            'total_silence_time_ms': 0.0,
            'turn_count': 0,
            'turn_durations': [],
            'interruptions_made': 0,
            'interruptions_received': 0,
            'overlaps_participated': 0,
            'volume_levels': []
        })
        
        # Audio metrics tracking
        self.volume_history = deque(maxlen=1000)
        self.pause_events: List[Dict[str, Any]] = []
        self.speaker_changes = 0
        self.interruption_events: List[Dict[str, Any]] = []
        self.overlap_events: List[Dict[str, Any]] = []
        
        # Rolling window data
        self.rolling_window_events = deque(maxlen=self.config.max_history_items)
        
        # Threading
        self.processing_thread = None
        self.thread_lock = threading.Lock()
        
        # Callbacks
        self.statistics_callbacks: List[Callable[[AudioStatistics], None]] = []
        self.speaker_change_callbacks: List[Callable[[str, Optional[str], float], None]] = []
        self.interruption_callbacks: List[Callable[[str, str, float], None]] = []
        
        # Statistics storage
        self.statistics_history: Dict[IntervalType, List[AudioStatistics]] = {
            interval: deque(maxlen=1000) for interval in IntervalType
        }
        
        logger.info("StatisticsCollector initialized")
    
    def add_statistics_callback(self, callback: Callable[[AudioStatistics], None]) -> None:
        """Add callback for statistics updates"""
        self.statistics_callbacks.append(callback)
    
    def add_speaker_change_callback(self, callback: Callable[[str, Optional[str], float], None]) -> None:
        """Add callback for speaker changes (from_speaker, to_speaker, timestamp)"""
        self.speaker_change_callbacks.append(callback)
    
    def add_interruption_callback(self, callback: Callable[[str, str, float], None]) -> None:
        """Add callback for interruptions (interrupting_speaker, interrupted_speaker, timestamp)"""
        self.interruption_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all callback lists"""
        removed = False
        for callback_list in [self.statistics_callbacks, self.speaker_change_callbacks, 
                             self.interruption_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    def _db_from_amplitude(self, amplitude: float) -> float:
        """Convert amplitude to dB scale"""
        if amplitude <= 0:
            return -float('inf')
        return 20 * np.log10(amplitude)
    
    def _detect_interruption(self, new_speaker: str, timestamp: float) -> Optional[str]:
        """Detect if this is an interruption"""
        # Check if another speaker was already talking recently
        recent_threshold = timestamp - (self.config.overlap_threshold_ms / 1000.0)
        
        for speaker_id, start_time in self.current_speakers.items():
            if speaker_id != new_speaker and start_time >= recent_threshold:
                # This looks like an interruption
                return speaker_id
        
        return None
    
    def _detect_overlap(self, timestamp: float) -> List[str]:
        """Detect concurrent speakers (overlapping speech)"""
        active_threshold = timestamp - (self.config.speaker_timeout_s / 2)  # More lenient for overlaps
        
        active_speakers = []
        for speaker_id, start_time in self.current_speakers.items():
            if start_time >= active_threshold:
                active_speakers.append(speaker_id)
        
        return active_speakers if len(active_speakers) > 1 else []
    
    def _calculate_speech_rate(self, speaking_time_ms: float, 
                             turn_count: int) -> float:
        """Estimate speech rate in words per minute"""
        if speaking_time_ms <= 0:
            return 0.0
        
        # Rough estimate: average speaker says 150 words per minute
        # Adjust based on turn patterns (more turns might indicate faster speech)
        base_wpm = 150.0
        speaking_minutes = speaking_time_ms / 60000.0
        
        if turn_count > 0 and speaking_minutes > 0:
            # Estimate based on turn frequency
            turns_per_minute = turn_count / speaking_minutes
            if turns_per_minute > 5:  # Very frequent turns might indicate faster speech
                base_wpm *= 1.2
            elif turns_per_minute < 1:  # Infrequent turns might indicate slower speech
                base_wpm *= 0.8
        
        return base_wpm
    
    def _calculate_participation_balance(self, speaking_stats: Dict[str, SpeakingTimeStats]) -> float:
        """Calculate participation balance score (0-1, higher = more balanced)"""
        if len(speaking_stats) <= 1:
            return 1.0  # Single speaker is perfectly "balanced" for themselves
        
        # Get speaking percentages
        percentages = [stats.speaking_percentage for stats in speaking_stats.values()]
        
        # Calculate coefficient of variation (lower = more balanced)
        if len(percentages) == 0:
            return 0.0
        
        mean_percentage = np.mean(percentages)
        if mean_percentage == 0:
            return 0.0
        
        std_percentage = np.std(percentages)
        cv = std_percentage / mean_percentage
        
        # Convert to balance score (lower CV = higher balance)
        # Perfect balance would have CV = 0, realistic good balance CV < 0.5
        balance_score = max(0.0, 1.0 - (cv / 1.0))  # Normalize by expected max CV
        
        return balance_score
    
    def update_speaker_activity(self, speaker_id: Optional[str], timestamp: Optional[float] = None,
                               voice_activity: bool = True, volume_db: Optional[float] = None,
                               confidence: float = 1.0) -> None:
        """
        Update speaker activity information
        
        Args:
            speaker_id: Current active speaker (None for silence)
            timestamp: Timestamp of the update
            voice_activity: Whether voice activity is detected
            volume_db: Current volume level in dB
            confidence: Confidence of speaker identification
        """
        if timestamp is None:
            timestamp = time.time()
        
        if volume_db is None:
            volume_db = -20.0  # Default reasonable level
        
        try:
            with self.thread_lock:
                if self.session_start_time is None:
                    self.session_start_time = timestamp
                
                # Track volume
                self.volume_history.append((timestamp, volume_db))
                
                # Handle speaker changes
                current_active = set(self.current_speakers.keys())
                
                if voice_activity and speaker_id:
                    # Voice activity with identified speaker
                    
                    # Check for interruptions
                    interrupted_speaker = self._detect_interruption(speaker_id, timestamp)
                    if interrupted_speaker:
                        # Record interruption
                        interruption_event = {
                            'timestamp': timestamp,
                            'interrupting_speaker': speaker_id,
                            'interrupted_speaker': interrupted_speaker,
                            'confidence': confidence
                        }
                        self.interruption_events.append(interruption_event)
                        self.speaker_stats[speaker_id]['interruptions_made'] += 1
                        self.speaker_stats[interrupted_speaker]['interruptions_received'] += 1
                        
                        # Fire interruption callbacks
                        for callback in self.interruption_callbacks:
                            try:
                                callback(speaker_id, interrupted_speaker, timestamp)
                            except Exception as e:
                                logger.error(f"Error in interruption callback: {e}")
                    
                    # Check for overlaps
                    overlapping_speakers = self._detect_overlap(timestamp)
                    if overlapping_speakers:
                        overlap_event = {
                            'timestamp': timestamp,
                            'speakers': overlapping_speakers,
                            'duration_ms': self.config.overlap_threshold_ms  # Estimated
                        }
                        self.overlap_events.append(overlap_event)
                        
                        for spk in overlapping_speakers:
                            self.speaker_stats[spk]['overlaps_participated'] += 1
                    
                    # Update current speakers
                    if speaker_id not in self.current_speakers:
                        # New speaker started
                        self.current_speakers[speaker_id] = timestamp
                        
                        # End previous speakers if timeout exceeded
                        to_remove = []
                        for spk_id, start_time in self.current_speakers.items():
                            if (spk_id != speaker_id and 
                                timestamp - start_time > self.config.speaker_timeout_s):
                                to_remove.append(spk_id)
                        
                        for spk_id in to_remove:
                            self._end_speaker_turn(spk_id, timestamp)
                        
                        # Record speaker change
                        previous_speaker = None
                        if current_active:
                            previous_speaker = list(current_active)[0]  # Simplified
                        
                        if previous_speaker != speaker_id:
                            self.speaker_changes += 1
                            
                            # Fire speaker change callbacks
                            for callback in self.speaker_change_callbacks:
                                try:
                                    callback(previous_speaker, speaker_id, timestamp)
                                except Exception as e:
                                    logger.error(f"Error in speaker change callback: {e}")
                        
                        # Create speaker event
                        event = SpeakerEvent(
                            timestamp=timestamp,
                            speaker_id=speaker_id,
                            event_type='start',
                            volume_db=volume_db,
                            confidence=confidence
                        )
                        self.speaker_events.append(event)
                        self.rolling_window_events.append(event)
                        
                        # Update turn count
                        self.speaker_stats[speaker_id]['turn_count'] += 1
                    
                    # Track volume for this speaker
                    self.speaker_stats[speaker_id]['volume_levels'].append(volume_db)
                
                else:
                    # No voice activity or no speaker identified
                    # End all current speakers
                    for speaker_id in list(self.current_speakers.keys()):
                        self._end_speaker_turn(speaker_id, timestamp)
                
                # Update timing if enough time has passed
                if (self.last_update_time is None or 
                    timestamp - self.last_update_time >= self.config.update_interval_s):
                    self._update_statistics(timestamp)
                    self.last_update_time = timestamp
                
        except Exception as e:
            logger.error(f"Error updating speaker activity: {e}")
    
    def _end_speaker_turn(self, speaker_id: str, end_time: float) -> None:
        """End a speaker's turn and update statistics"""
        if speaker_id not in self.current_speakers:
            return
        
        start_time = self.current_speakers[speaker_id]
        duration_ms = (end_time - start_time) * 1000
        
        # Update speaking time
        self.speaker_stats[speaker_id]['total_speaking_time_ms'] += duration_ms
        self.speaker_stats[speaker_id]['turn_durations'].append(duration_ms)
        
        # Create end event
        event = SpeakerEvent(
            timestamp=end_time,
            speaker_id=speaker_id,
            event_type='end',
            volume_db=0.0,  # Will be updated with actual volume if available
            confidence=1.0,
            duration_ms=duration_ms
        )
        self.speaker_events.append(event)
        self.rolling_window_events.append(event)
        
        # Remove from current speakers
        del self.current_speakers[speaker_id]
    
    def _update_statistics(self, timestamp: float) -> None:
        """Update and calculate current statistics"""
        try:
            # Calculate session duration
            session_duration_ms = 0.0
            if self.session_start_time:
                session_duration_ms = (timestamp - self.session_start_time) * 1000
            
            # Calculate speaking statistics for each speaker
            speaking_stats = {}
            total_speaking_time = 0.0
            
            for speaker_id, stats in self.speaker_stats.items():
                turn_durations = stats['turn_durations']
                
                # Calculate turn statistics
                avg_turn_duration = np.mean(turn_durations) if turn_durations else 0.0
                longest_turn = max(turn_durations) if turn_durations else 0.0
                shortest_turn = min(turn_durations) if turn_durations else 0.0
                
                # Calculate speaking percentage
                speaking_percentage = 0.0
                if session_duration_ms > 0:
                    speaking_percentage = (stats['total_speaking_time_ms'] / session_duration_ms) * 100
                
                total_speaking_time += stats['total_speaking_time_ms']
                
                speaking_stats[speaker_id] = SpeakingTimeStats(
                    speaker_id=speaker_id,
                    total_speaking_time_ms=stats['total_speaking_time_ms'],
                    total_silence_time_ms=session_duration_ms - stats['total_speaking_time_ms'],
                    speaking_percentage=speaking_percentage,
                    turn_count=stats['turn_count'],
                    average_turn_duration_ms=avg_turn_duration,
                    longest_turn_ms=longest_turn,
                    shortest_turn_ms=shortest_turn,
                    interruptions_made=stats['interruptions_made'],
                    interruptions_received=stats['interruptions_received'],
                    overlaps_participated=stats['overlaps_participated']
                )
            
            # Calculate volume statistics
            recent_volumes = [vol for ts, vol in self.volume_history 
                            if timestamp - ts <= 60.0]  # Last minute
            
            avg_volume = np.mean(recent_volumes) if recent_volumes else -40.0
            peak_volume = max(recent_volumes) if recent_volumes else -40.0
            volume_variance = np.var(recent_volumes) if recent_volumes else 0.0
            
            # Calculate participation metrics
            active_speakers = len([s for s in speaking_stats.values() 
                                 if s.speaking_percentage > 5.0])  # >5% participation
            
            # Find dominant speaker
            dominant_speaker = None
            if speaking_stats:
                dominant_speaker = max(speaking_stats.keys(),
                                     key=lambda x: speaking_stats[x].speaking_percentage)
            
            # Calculate rates
            session_minutes = session_duration_ms / 60000.0 if session_duration_ms > 0 else 1.0
            turn_taking_frequency = self.speaker_changes / session_minutes
            interruption_rate = len(self.interruption_events) / session_minutes
            
            # Calculate average turn gap (simplified)
            avg_turn_gap = 1000.0  # Default 1 second
            if len(self.speaker_events) > 1:
                gaps = []
                for i in range(1, len(self.speaker_events)):
                    if (self.speaker_events[i].event_type == 'start' and
                        self.speaker_events[i-1].event_type == 'end'):
                        gap = (self.speaker_events[i].timestamp - 
                              self.speaker_events[i-1].timestamp) * 1000
                        gaps.append(gap)
                
                if gaps:
                    avg_turn_gap = np.mean(gaps)
            
            # Calculate participation balance
            balance_score = self._calculate_participation_balance(speaking_stats)
            
            # Create participation metrics
            participation_metrics = ParticipationMetrics(
                total_speakers=len(speaking_stats),
                active_speakers=active_speakers,
                total_session_time_ms=session_duration_ms,
                total_speaking_time_ms=total_speaking_time,
                total_silence_time_ms=session_duration_ms - total_speaking_time,
                speaking_ratio=total_speaking_time / session_duration_ms if session_duration_ms > 0 else 0.0,
                silence_ratio=(session_duration_ms - total_speaking_time) / session_duration_ms if session_duration_ms > 0 else 1.0,
                dominant_speaker=dominant_speaker,
                most_balanced_period_start_ms=None,  # Could be calculated with more complex analysis
                most_balanced_period_duration_ms=None,
                turn_taking_frequency=turn_taking_frequency,
                average_turn_gap_ms=avg_turn_gap,
                interruption_rate=interruption_rate,
                participation_balance_score=balance_score
            )
            
            # Calculate speech rate (simplified)
            total_turns = sum(stats.turn_count for stats in speaking_stats.values())
            speech_rate = self._calculate_speech_rate(total_speaking_time, total_turns)
            
            # Calculate pause metrics
            pause_frequency = len(self.pause_events) / session_minutes
            avg_pause_duration = np.mean([p['duration_ms'] for p in self.pause_events]) if self.pause_events else 0.0
            
            # Create comprehensive statistics
            audio_stats = AudioStatistics(
                timestamp=timestamp,
                interval_type=IntervalType.REAL_TIME,
                interval_duration_ms=session_duration_ms,
                speaking_stats=speaking_stats,
                participation_metrics=participation_metrics,
                average_volume_db=avg_volume,
                peak_volume_db=peak_volume,
                volume_variance=volume_variance,
                signal_to_noise_ratio=avg_volume + 40,  # Simplified SNR estimate
                speech_rate_wpm=speech_rate,
                pause_frequency=pause_frequency,
                average_pause_duration_ms=avg_pause_duration,
                concurrent_speakers=len(self.current_speakers),
                speaker_changes=self.speaker_changes,
                total_interruptions=len(self.interruption_events),
                total_overlaps=len(self.overlap_events)
            )
            
            # Store statistics
            self.statistics_history[IntervalType.REAL_TIME].append(audio_stats)
            
            # Fire callbacks
            for callback in self.statistics_callbacks:
                try:
                    callback(audio_stats)
                except Exception as e:
                    logger.error(f"Error in statistics callback: {e}")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def get_current_statistics(self) -> Optional[AudioStatistics]:
        """Get the most recent statistics"""
        with self.thread_lock:
            if self.statistics_history[IntervalType.REAL_TIME]:
                return self.statistics_history[IntervalType.REAL_TIME][-1]
            return None
    
    def get_statistics_history(self, interval_type: IntervalType = IntervalType.REAL_TIME,
                              limit: Optional[int] = None) -> List[AudioStatistics]:
        """Get statistics history for specified interval"""
        with self.thread_lock:
            history = list(self.statistics_history[interval_type])
            if limit:
                history = history[-limit:]
            return history
    
    def get_speaker_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all speakers"""
        with self.thread_lock:
            current_stats = self.get_current_statistics()
            if not current_stats:
                return {}
            
            summary = {}
            for speaker_id, stats in current_stats.speaking_stats.items():
                summary[speaker_id] = {
                    'total_speaking_time_minutes': stats.total_speaking_time_ms / 60000.0,
                    'speaking_percentage': stats.speaking_percentage,
                    'turn_count': stats.turn_count,
                    'average_turn_duration_seconds': stats.average_turn_duration_ms / 1000.0,
                    'interruptions_made': stats.interruptions_made,
                    'interruptions_received': stats.interruptions_received,
                    'participation_level': self._classify_participation(stats.speaking_percentage)
                }
            
            return summary
    
    def _classify_participation(self, speaking_percentage: float) -> str:
        """Classify participation level based on speaking percentage"""
        if speaking_percentage >= 40:
            return "Very High"
        elif speaking_percentage >= 25:
            return "High"
        elif speaking_percentage >= 15:
            return "Medium"
        elif speaking_percentage >= 5:
            return "Low"
        else:
            return "Very Low"
    
    def export_statistics(self, format_type: str = 'json') -> str:
        """Export statistics in specified format"""
        with self.thread_lock:
            current_stats = self.get_current_statistics()
            if not current_stats:
                return ""
            
            if format_type.lower() == 'json':
                return json.dumps(current_stats.to_dict(), indent=2)
            elif format_type.lower() == 'csv':
                # Simplified CSV export
                lines = ["speaker_id,speaking_time_ms,speaking_percentage,turn_count,interruptions"]
                for speaker_id, stats in current_stats.speaking_stats.items():
                    lines.append(f"{speaker_id},{stats.total_speaking_time_ms},"
                               f"{stats.speaking_percentage},{stats.turn_count},"
                               f"{stats.interruptions_made + stats.interruptions_received}")
                return "\n".join(lines)
            else:
                return str(current_stats.to_dict())
    
    def reset_statistics(self) -> None:
        """Reset all statistics and start fresh"""
        with self.thread_lock:
            self.session_start_time = None
            self.last_update_time = None
            self.current_speakers.clear()
            self.speaker_events.clear()
            self.speaker_stats.clear()
            self.volume_history.clear()
            self.pause_events.clear()
            self.speaker_changes = 0
            self.interruption_events.clear()
            self.overlap_events.clear()
            self.rolling_window_events.clear()
            
            for interval_type in IntervalType:
                self.statistics_history[interval_type].clear()
        
        logger.info("Statistics collector reset")
    
    def start_processing(self) -> bool:
        """Start statistics collection"""
        if self.is_running:
            logger.warning("Statistics collection already running")
            return False
        
        with self.thread_lock:
            self.is_running = True
            self.session_start_time = time.time()
        
        logger.info("Statistics collection started")
        return True
    
    def stop_processing(self) -> bool:
        """Stop statistics collection"""
        if not self.is_running:
            logger.warning("Statistics collection not running")
            return False
        
        with self.thread_lock:
            self.is_running = False
            
            # End all ongoing speaker turns
            current_time = time.time()
            for speaker_id in list(self.current_speakers.keys()):
                self._end_speaker_turn(speaker_id, current_time)
            
            # Final statistics update
            self._update_statistics(current_time)
        
        logger.info("Statistics collection stopped")
        return True
    
    def is_processing(self) -> bool:
        """Check if statistics collection is running"""
        return self.is_running

def create_statistics_collector(config: Optional[StatisticsConfig] = None) -> StatisticsCollector:
    """
    Factory function to create a configured statistics collector
    
    Args:
        config: Optional statistics collection configuration
        
    Returns:
        Configured StatisticsCollector instance
    """
    return StatisticsCollector(config)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create statistics collector
    config = StatisticsConfig(
        update_interval_ms=2000,  # Update every 2 seconds
        speaker_timeout_ms=1000   # 1 second timeout
    )
    
    collector = create_statistics_collector(config)
    
    # Add example callbacks
    def on_statistics_update(stats: AudioStatistics):
        print(f"\nStatistics Update:")
        print(f"  Session duration: {stats.interval_duration_ms / 1000:.1f}s")
        print(f"  Active speakers: {stats.participation_metrics.active_speakers}")
        print(f"  Speaker changes: {stats.speaker_changes}")
        print(f"  Balance score: {stats.participation_metrics.participation_balance_score:.2f}")
        
        for speaker_id, speaker_stats in stats.speaking_stats.items():
            print(f"  {speaker_id}: {speaker_stats.speaking_percentage:.1f}% "
                  f"({speaker_stats.turn_count} turns)")
    
    def on_speaker_change(from_speaker: Optional[str], to_speaker: str, timestamp: float):
        print(f"Speaker change: {from_speaker} -> {to_speaker}")
    
    def on_interruption(interrupting: str, interrupted: str, timestamp: float):
        print(f"Interruption: {interrupting} interrupted {interrupted}")
    
    collector.add_statistics_callback(on_statistics_update)
    collector.add_speaker_change_callback(on_speaker_change)
    collector.add_interruption_callback(on_interruption)
    
    # Start processing
    collector.start_processing()
    
    # Test with simulated speaker activity
    try:
        print("Testing statistics collector with simulated meeting...")
        
        speakers = ["Alice", "Bob", "Charlie"]
        
        # Simulate a 30-second meeting
        for i in range(30):
            # Simulate different speaking patterns
            if i < 10:
                # Alice speaks first
                active_speaker = "Alice"
                volume = -15.0
            elif i < 15:
                # Bob interrupts
                active_speaker = "Bob"
                volume = -12.0
            elif i < 20:
                # Alice responds
                active_speaker = "Alice"
                volume = -14.0
            elif i < 25:
                # Charlie joins
                active_speaker = "Charlie"
                volume = -18.0
            else:
                # Group discussion (simulate overlaps)
                active_speaker = speakers[i % 3]
                volume = -16.0
            
            # Add some silence periods
            if i % 7 == 0:
                active_speaker = None
                voice_activity = False
            else:
                voice_activity = True
            
            collector.update_speaker_activity(
                speaker_id=active_speaker,
                voice_activity=voice_activity,
                volume_db=volume,
                confidence=0.8
            )
            
            time.sleep(1.0)  # 1 second intervals
        
        # Get final summary
        print(f"\nFinal Summary:")
        summary = collector.get_speaker_summary()
        for speaker_id, stats in summary.items():
            print(f"{speaker_id}:")
            print(f"  Speaking time: {stats['total_speaking_time_minutes']:.1f} minutes")
            print(f"  Speaking percentage: {stats['speaking_percentage']:.1f}%")
            print(f"  Turns: {stats['turn_count']}")
            print(f"  Interruptions: {stats['interruptions_made']} made, "
                  f"{stats['interruptions_received']} received")
            print(f"  Participation: {stats['participation_level']}")
        
        # Export statistics
        print(f"\nExporting statistics...")
        json_export = collector.export_statistics('json')
        print(f"JSON export length: {len(json_export)} characters")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        collector.stop_processing()
        print("Statistics collection test completed")