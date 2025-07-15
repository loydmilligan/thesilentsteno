#!/usr/bin/env python3
"""
Transcript Formatter Module

Multiple format transcript output with speaker labels, timestamps, and export options.
Supports various output formats for different use cases.

Author: Claude AI Assistant
Date: 2024-07-14
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta
import re

try:
    import numpy as np
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    XML = "xml"


class TimestampFormat(Enum):
    """Timestamp format options"""
    SECONDS = "seconds"  # 123.45
    MILLISECONDS = "milliseconds"  # 123450
    HOURS_MINUTES_SECONDS = "hms"  # 02:03:45
    HOURS_MINUTES_SECONDS_MS = "hms_ms"  # 02:03:45.123
    SRT_FORMAT = "srt"  # 00:02:03,456
    VTT_FORMAT = "vtt"  # 00:02:03.456


@dataclass
class FormattingConfig:
    """Configuration for transcript formatting"""
    
    # Output format
    output_format: OutputFormat = OutputFormat.TEXT
    
    # Timestamp settings
    include_timestamps: bool = True
    timestamp_format: TimestampFormat = TimestampFormat.HOURS_MINUTES_SECONDS
    timestamp_precision: int = 3  # decimal places
    
    # Speaker settings
    include_speakers: bool = True
    speaker_format: str = "{label}:"  # Format for speaker labels
    anonymous_speakers: bool = False  # Use generic labels
    
    # Content settings
    include_confidence: bool = False
    confidence_threshold: float = 0.7
    filter_low_confidence: bool = True
    
    # Text formatting
    capitalize_sentences: bool = True
    add_punctuation: bool = True
    normalize_whitespace: bool = True
    line_break_on_speaker_change: bool = True
    
    # Segmentation
    max_line_length: int = 80
    max_segment_duration: float = 30.0  # seconds
    merge_short_segments: bool = True
    min_segment_duration: float = 0.5  # seconds
    
    # Export settings
    include_metadata: bool = True
    include_statistics: bool = True
    
    # Language settings
    language: str = "en"
    encoding: str = "utf-8"
    
    # Template settings
    custom_template: Optional[str] = None
    template_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranscriptSegment:
    """Individual transcript segment"""
    
    # Core content
    text: str
    speaker_id: int
    speaker_label: str
    
    # Timing
    start_time: float
    end_time: float
    duration: float
    
    # Quality
    confidence: float
    voice_activity: float
    
    # Metadata
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    word_count: int = 0
    character_count: int = 0
    
    # Processing info
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.text:
            self.word_count = len(self.text.split())
            self.character_count = len(self.text)


@dataclass
class TranscriptMetadata:
    """Transcript metadata"""
    
    # Basic info
    title: str = ""
    description: str = ""
    language: str = "en"
    
    # Timing
    total_duration: float = 0.0
    recording_date: Optional[datetime] = None
    created_date: datetime = field(default_factory=datetime.now)
    
    # Participants
    speaker_count: int = 0
    speaker_labels: List[str] = field(default_factory=list)
    
    # Quality metrics
    average_confidence: float = 0.0
    total_segments: int = 0
    total_words: int = 0
    total_characters: int = 0
    
    # Processing info
    processing_time: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    # Export info
    export_format: str = "text"
    export_date: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class FormattedTranscript:
    """Complete formatted transcript"""
    
    # Content
    content: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    metadata: TranscriptMetadata = field(default_factory=TranscriptMetadata)
    
    # Format info
    format_type: OutputFormat = OutputFormat.TEXT
    config: FormattingConfig = field(default_factory=FormattingConfig)
    
    # Export info
    file_path: Optional[str] = None
    file_size: int = 0
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "segments": [seg.__dict__ for seg in self.segments],
            "metadata": self.metadata.__dict__,
            "format_type": self.format_type.value,
            "config": self.config.__dict__,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "statistics": self.statistics
        }


class TranscriptFormatter:
    """Main transcript formatting engine"""
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        self.config = config or FormattingConfig()
        self.is_initialized = False
        
        # Statistics
        self.stats = {
            "total_formatted": 0,
            "total_segments_processed": 0,
            "total_processing_time": 0.0,
            "formats_generated": {},
            "average_confidence": 0.0
        }
        
        # Processing state
        self.processing_lock = threading.Lock()
        
        logger.info(f"TranscriptFormatter initialized with {self.config.output_format.value} format")
        
    def initialize(self) -> bool:
        """Initialize the formatter"""
        try:
            logger.info("Initializing transcript formatter...")
            
            # Check for optional dependencies
            self._check_dependencies()
            
            self.is_initialized = True
            logger.info("Transcript formatter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize formatter: {e}")
            return False
            
    def _check_dependencies(self):
        """Check for format-specific dependencies"""
        try:
            # Check for DOCX support
            if self.config.output_format == OutputFormat.DOCX:
                try:
                    import docx
                    logger.info("DOCX support available")
                except ImportError:
                    logger.warning("DOCX support not available. Install python-docx")
                    
            # Check for PDF support
            if self.config.output_format == OutputFormat.PDF:
                try:
                    import reportlab
                    logger.info("PDF support available")
                except ImportError:
                    logger.warning("PDF support not available. Install reportlab")
                    
        except Exception as e:
            logger.warning(f"Error checking dependencies: {e}")
            
    def format_transcript(self, segments: List[TranscriptSegment], 
                         metadata: Optional[TranscriptMetadata] = None) -> FormattedTranscript:
        """Format transcript segments into specified format"""
        if not self.is_initialized:
            logger.error("Formatter not initialized")
            return FormattedTranscript(content="", format_type=self.config.output_format)
            
        start_time = time.time()
        
        try:
            # Process segments
            processed_segments = self._process_segments(segments)
            
            # Create metadata if not provided
            if metadata is None:
                metadata = self._create_metadata(processed_segments)
                
            # Format content
            content = self._format_content(processed_segments, metadata)
            
            # Calculate statistics
            statistics = self._calculate_statistics(processed_segments)
            
            # Create formatted transcript
            formatted_transcript = FormattedTranscript(
                content=content,
                segments=processed_segments,
                metadata=metadata,
                format_type=self.config.output_format,
                config=self.config,
                statistics=statistics
            )
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self._update_stats(formatted_transcript, processing_time)
            
            logger.info(f"Formatted {len(processed_segments)} segments in {processing_time:.3f}s")
            return formatted_transcript
            
        except Exception as e:
            logger.error(f"Failed to format transcript: {e}")
            return FormattedTranscript(content="", format_type=self.config.output_format)
            
    def _process_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Process and clean segments"""
        processed_segments = []
        
        for segment in segments:
            # Skip low confidence segments if configured
            if (self.config.filter_low_confidence and 
                segment.confidence < self.config.confidence_threshold):
                continue
                
            # Process text
            text = segment.text
            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
                
            if self.config.capitalize_sentences:
                text = self._capitalize_sentences(text)
                
            if self.config.add_punctuation:
                text = self._add_punctuation(text)
                
            # Update segment
            processed_segment = TranscriptSegment(
                text=text,
                speaker_id=segment.speaker_id,
                speaker_label=self._format_speaker_label(segment.speaker_label),
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                confidence=segment.confidence,
                voice_activity=segment.voice_activity,
                segment_id=segment.segment_id
            )
            
            processed_segments.append(processed_segment)
            
        # Merge short segments if configured
        if self.config.merge_short_segments:
            processed_segments = self._merge_short_segments(processed_segments)
            
        return processed_segments
        
    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of sentences"""
        if not text:
            return text
            
        # Simple sentence capitalization
        sentences = re.split(r'([.!?]+)', text)
        result = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0:  # Text (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            result.append(sentence)
            
        return ''.join(result)
        
    def _add_punctuation(self, text: str) -> str:
        """Add basic punctuation if missing"""
        if not text:
            return text
            
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
            
        return text
        
    def _format_speaker_label(self, label: str) -> str:
        """Format speaker label according to config"""
        if self.config.anonymous_speakers:
            # Extract speaker number and use generic format
            match = re.search(r'(\d+)', label)
            if match:
                return f"Speaker {match.group(1)}"
            return "Speaker"
            
        return self.config.speaker_format.format(label=label)
        
    def _merge_short_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Merge consecutive short segments from same speaker"""
        if not segments:
            return segments
            
        merged_segments = []
        current_segment = None
        
        for segment in segments:
            if (current_segment is None or 
                current_segment.speaker_id != segment.speaker_id or
                current_segment.duration >= self.config.max_segment_duration):
                # Start new segment
                if current_segment:
                    merged_segments.append(current_segment)
                current_segment = segment
            else:
                # Merge with current segment
                current_segment.text += " " + segment.text
                current_segment.end_time = segment.end_time
                current_segment.duration = current_segment.end_time - current_segment.start_time
                current_segment.confidence = max(current_segment.confidence, segment.confidence)
                current_segment.voice_activity = max(current_segment.voice_activity, segment.voice_activity)
                
        if current_segment:
            merged_segments.append(current_segment)
            
        return merged_segments
        
    def _create_metadata(self, segments: List[TranscriptSegment]) -> TranscriptMetadata:
        """Create metadata from segments"""
        if not segments:
            return TranscriptMetadata()
            
        # Calculate statistics
        total_duration = segments[-1].end_time - segments[0].start_time if segments else 0.0
        speaker_ids = list(set(seg.speaker_id for seg in segments))
        speaker_labels = list(set(seg.speaker_label for seg in segments))
        average_confidence = np.mean([seg.confidence for seg in segments])
        total_words = sum(seg.word_count for seg in segments)
        total_characters = sum(seg.character_count for seg in segments)
        
        return TranscriptMetadata(
            language=self.config.language,
            total_duration=total_duration,
            speaker_count=len(speaker_ids),
            speaker_labels=speaker_labels,
            average_confidence=average_confidence,
            total_segments=len(segments),
            total_words=total_words,
            total_characters=total_characters,
            export_format=self.config.output_format.value
        )
        
    def _format_content(self, segments: List[TranscriptSegment], 
                       metadata: TranscriptMetadata) -> str:
        """Format content according to output format"""
        format_handlers = {
            OutputFormat.TEXT: self._format_text,
            OutputFormat.JSON: self._format_json,
            OutputFormat.SRT: self._format_srt,
            OutputFormat.VTT: self._format_vtt,
            OutputFormat.HTML: self._format_html,
            OutputFormat.MARKDOWN: self._format_markdown,
            OutputFormat.CSV: self._format_csv,
            OutputFormat.XML: self._format_xml
        }
        
        handler = format_handlers.get(self.config.output_format, self._format_text)
        return handler(segments, metadata)
        
    def _format_text(self, segments: List[TranscriptSegment], 
                    metadata: TranscriptMetadata) -> str:
        """Format as plain text"""
        lines = []
        
        # Add header if metadata included
        if self.config.include_metadata:
            lines.append(f"Transcript - {metadata.created_date.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"Duration: {self._format_duration(metadata.total_duration)}")
            lines.append(f"Speakers: {metadata.speaker_count}")
            lines.append("")
            
        # Add content
        current_speaker = None
        for segment in segments:
            # Add speaker change line break
            if (self.config.line_break_on_speaker_change and 
                segment.speaker_id != current_speaker):
                if current_speaker is not None:
                    lines.append("")
                current_speaker = segment.speaker_id
                
            # Format line
            line_parts = []
            
            if self.config.include_timestamps:
                timestamp = self._format_timestamp(segment.start_time)
                line_parts.append(f"[{timestamp}]")
                
            if self.config.include_speakers:
                line_parts.append(segment.speaker_label)
                
            line_parts.append(segment.text)
            
            if self.config.include_confidence:
                line_parts.append(f"({segment.confidence:.2f})")
                
            line = " ".join(line_parts)
            
            # Wrap long lines
            if len(line) > self.config.max_line_length:
                line = self._wrap_line(line, self.config.max_line_length)
                
            lines.append(line)
            
        return "\n".join(lines)
        
    def _format_json(self, segments: List[TranscriptSegment], 
                    metadata: TranscriptMetadata) -> str:
        """Format as JSON"""
        data = {
            "metadata": metadata.__dict__,
            "segments": []
        }
        
        for segment in segments:
            seg_data = {
                "id": segment.segment_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "speaker_id": segment.speaker_id,
                "speaker_label": segment.speaker_label,
                "text": segment.text,
                "confidence": segment.confidence,
                "voice_activity": segment.voice_activity,
                "word_count": segment.word_count,
                "character_count": segment.character_count
            }
            data["segments"].append(seg_data)
            
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
    def _format_srt(self, segments: List[TranscriptSegment], 
                   metadata: TranscriptMetadata) -> str:
        """Format as SRT subtitles"""
        lines = []
        
        for i, segment in enumerate(segments, 1):
            # Subtitle number
            lines.append(str(i))
            
            # Timestamp
            start_time = self._format_timestamp(segment.start_time, TimestampFormat.SRT_FORMAT)
            end_time = self._format_timestamp(segment.end_time, TimestampFormat.SRT_FORMAT)
            lines.append(f"{start_time} --> {end_time}")
            
            # Text with speaker
            if self.config.include_speakers:
                text = f"{segment.speaker_label} {segment.text}"
            else:
                text = segment.text
                
            lines.append(text)
            lines.append("")  # Empty line between subtitles
            
        return "\n".join(lines)
        
    def _format_vtt(self, segments: List[TranscriptSegment], 
                   metadata: TranscriptMetadata) -> str:
        """Format as WebVTT"""
        lines = ["WEBVTT", ""]
        
        for segment in segments:
            # Timestamp
            start_time = self._format_timestamp(segment.start_time, TimestampFormat.VTT_FORMAT)
            end_time = self._format_timestamp(segment.end_time, TimestampFormat.VTT_FORMAT)
            lines.append(f"{start_time} --> {end_time}")
            
            # Text with speaker
            if self.config.include_speakers:
                text = f"<v {segment.speaker_label}>{segment.text}"
            else:
                text = segment.text
                
            lines.append(text)
            lines.append("")  # Empty line between cues
            
        return "\n".join(lines)
        
    def _format_html(self, segments: List[TranscriptSegment], 
                    metadata: TranscriptMetadata) -> str:
        """Format as HTML"""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Transcript - {metadata.created_date.strftime('%Y-%m-%d')}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; }",
            ".transcript { max-width: 800px; margin: 0 auto; }",
            ".segment { margin: 10px 0; }",
            ".timestamp { color: #666; font-size: 0.9em; }",
            ".speaker { font-weight: bold; color: #333; }",
            ".confidence { color: #999; font-size: 0.8em; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='transcript'>"
        ]
        
        # Add metadata
        if self.config.include_metadata:
            lines.extend([
                f"<h1>Transcript</h1>",
                f"<p>Date: {metadata.created_date.strftime('%Y-%m-%d %H:%M')}</p>",
                f"<p>Duration: {self._format_duration(metadata.total_duration)}</p>",
                f"<p>Speakers: {metadata.speaker_count}</p>",
                "<hr>"
            ])
            
        # Add segments
        for segment in segments:
            lines.append("<div class='segment'>")
            
            if self.config.include_timestamps:
                timestamp = self._format_timestamp(segment.start_time)
                lines.append(f"<span class='timestamp'>[{timestamp}]</span> ")
                
            if self.config.include_speakers:
                lines.append(f"<span class='speaker'>{segment.speaker_label}</span> ")
                
            lines.append(segment.text)
            
            if self.config.include_confidence:
                lines.append(f" <span class='confidence'>({segment.confidence:.2f})</span>")
                
            lines.append("</div>")
            
        lines.extend([
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(lines)
        
    def _format_markdown(self, segments: List[TranscriptSegment], 
                        metadata: TranscriptMetadata) -> str:
        """Format as Markdown"""
        lines = []
        
        # Add metadata
        if self.config.include_metadata:
            lines.extend([
                "# Transcript",
                "",
                f"**Date:** {metadata.created_date.strftime('%Y-%m-%d %H:%M')}",
                f"**Duration:** {self._format_duration(metadata.total_duration)}",
                f"**Speakers:** {metadata.speaker_count}",
                "",
                "---",
                ""
            ])
            
        # Add segments
        current_speaker = None
        for segment in segments:
            # Add speaker header
            if (self.config.line_break_on_speaker_change and 
                segment.speaker_id != current_speaker):
                if current_speaker is not None:
                    lines.append("")
                current_speaker = segment.speaker_id
                
            # Format line
            line_parts = []
            
            if self.config.include_timestamps:
                timestamp = self._format_timestamp(segment.start_time)
                line_parts.append(f"*[{timestamp}]*")
                
            if self.config.include_speakers:
                line_parts.append(f"**{segment.speaker_label}**")
                
            line_parts.append(segment.text)
            
            if self.config.include_confidence:
                line_parts.append(f"*({segment.confidence:.2f})*")
                
            lines.append(" ".join(line_parts))
            
        return "\n".join(lines)
        
    def _format_csv(self, segments: List[TranscriptSegment], 
                   metadata: TranscriptMetadata) -> str:
        """Format as CSV"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        header = ["start_time", "end_time", "duration", "speaker_id", "speaker_label", "text"]
        if self.config.include_confidence:
            header.append("confidence")
        writer.writerow(header)
        
        # Data rows
        for segment in segments:
            row = [
                segment.start_time,
                segment.end_time,
                segment.duration,
                segment.speaker_id,
                segment.speaker_label,
                segment.text
            ]
            if self.config.include_confidence:
                row.append(segment.confidence)
            writer.writerow(row)
            
        return output.getvalue()
        
    def _format_xml(self, segments: List[TranscriptSegment], 
                   metadata: TranscriptMetadata) -> str:
        """Format as XML"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<transcript>'
        ]
        
        # Add metadata
        if self.config.include_metadata:
            lines.extend([
                '  <metadata>',
                f'    <created_date>{metadata.created_date.isoformat()}</created_date>',
                f'    <duration>{metadata.total_duration}</duration>',
                f'    <speaker_count>{metadata.speaker_count}</speaker_count>',
                f'    <total_segments>{metadata.total_segments}</total_segments>',
                '  </metadata>'
            ])
            
        # Add segments
        lines.append('  <segments>')
        for segment in segments:
            lines.extend([
                '    <segment>',
                f'      <id>{segment.segment_id}</id>',
                f'      <start_time>{segment.start_time}</start_time>',
                f'      <end_time>{segment.end_time}</end_time>',
                f'      <speaker_id>{segment.speaker_id}</speaker_id>',
                f'      <speaker_label>{segment.speaker_label}</speaker_label>',
                f'      <text>{segment.text}</text>',
                f'      <confidence>{segment.confidence}</confidence>',
                '    </segment>'
            ])
        lines.append('  </segments>')
        
        lines.append('</transcript>')
        
        return "\n".join(lines)
        
    def _format_timestamp(self, seconds: float, 
                         format_type: Optional[TimestampFormat] = None) -> str:
        """Format timestamp according to specified format"""
        if format_type is None:
            format_type = self.config.timestamp_format
            
        if format_type == TimestampFormat.SECONDS:
            return f"{seconds:.{self.config.timestamp_precision}f}"
        elif format_type == TimestampFormat.MILLISECONDS:
            return str(int(seconds * 1000))
        elif format_type == TimestampFormat.HOURS_MINUTES_SECONDS:
            return str(timedelta(seconds=seconds)).split('.')[0]
        elif format_type == TimestampFormat.HOURS_MINUTES_SECONDS_MS:
            td = timedelta(seconds=seconds)
            return f"{str(td).split('.')[0]}.{int(td.microseconds/1000):03d}"
        elif format_type == TimestampFormat.SRT_FORMAT:
            td = timedelta(seconds=seconds)
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            milliseconds = int(td.microseconds / 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        elif format_type == TimestampFormat.VTT_FORMAT:
            td = timedelta(seconds=seconds)
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            milliseconds = int(td.microseconds / 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        else:
            return str(seconds)
            
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        return str(timedelta(seconds=seconds)).split('.')[0]
        
    def _wrap_line(self, line: str, max_length: int) -> str:
        """Wrap long lines"""
        if len(line) <= max_length:
            return line
            
        # Simple word wrapping
        words = line.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Single word too long
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += len(word) + 1
                
        if current_line:
            lines.append(" ".join(current_line))
            
        return "\n".join(lines)
        
    def _calculate_statistics(self, segments: List[TranscriptSegment]) -> Dict[str, Any]:
        """Calculate transcript statistics"""
        if not segments:
            return {}
            
        return {
            "total_segments": len(segments),
            "total_duration": segments[-1].end_time - segments[0].start_time,
            "total_words": sum(seg.word_count for seg in segments),
            "total_characters": sum(seg.character_count for seg in segments),
            "average_confidence": np.mean([seg.confidence for seg in segments]),
            "average_segment_duration": np.mean([seg.duration for seg in segments]),
            "speaker_count": len(set(seg.speaker_id for seg in segments)),
            "words_per_minute": sum(seg.word_count for seg in segments) / (
                (segments[-1].end_time - segments[0].start_time) / 60
            ) if segments else 0
        }
        
    def _update_stats(self, formatted_transcript: FormattedTranscript, processing_time: float):
        """Update formatting statistics"""
        with self.processing_lock:
            self.stats["total_formatted"] += 1
            self.stats["total_segments_processed"] += len(formatted_transcript.segments)
            self.stats["total_processing_time"] += processing_time
            
            # Update format counts
            format_name = formatted_transcript.format_type.value
            self.stats["formats_generated"][format_name] = (
                self.stats["formats_generated"].get(format_name, 0) + 1
            )
            
            # Update average confidence
            if formatted_transcript.segments:
                avg_confidence = np.mean([seg.confidence for seg in formatted_transcript.segments])
                count = self.stats["total_formatted"]
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (old_avg * (count - 1) + avg_confidence) / count
                
    def export_to_file(self, formatted_transcript: FormattedTranscript, 
                      file_path: str) -> bool:
        """Export formatted transcript to file"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=self.config.encoding) as f:
                f.write(formatted_transcript.content)
                
            # Update transcript info
            formatted_transcript.file_path = file_path
            formatted_transcript.file_size = os.path.getsize(file_path)
            
            logger.info(f"Exported transcript to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export transcript: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get formatting statistics"""
        with self.processing_lock:
            return self.stats.copy()
            
    def get_status(self) -> Dict[str, Any]:
        """Get formatter status"""
        return {
            "is_initialized": self.is_initialized,
            "output_format": self.config.output_format.value,
            "timestamp_format": self.config.timestamp_format.value,
            "include_speakers": self.config.include_speakers,
            "include_timestamps": self.config.include_timestamps,
            "include_confidence": self.config.include_confidence,
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown formatter"""
        logger.info("Shutting down transcript formatter...")
        self.is_initialized = False
        logger.info("Transcript formatter shutdown complete")


# Factory functions
def create_text_formatter(include_timestamps: bool = True, 
                         include_speakers: bool = True) -> TranscriptFormatter:
    """Create a text formatter with common settings"""
    config = FormattingConfig(
        output_format=OutputFormat.TEXT,
        include_timestamps=include_timestamps,
        include_speakers=include_speakers,
        timestamp_format=TimestampFormat.HOURS_MINUTES_SECONDS,
        line_break_on_speaker_change=True
    )
    return TranscriptFormatter(config)


def create_subtitle_formatter(subtitle_format: str = "srt") -> TranscriptFormatter:
    """Create a subtitle formatter"""
    format_map = {
        "srt": OutputFormat.SRT,
        "vtt": OutputFormat.VTT
    }
    
    config = FormattingConfig(
        output_format=format_map.get(subtitle_format, OutputFormat.SRT),
        include_speakers=True,
        include_timestamps=True,
        max_segment_duration=5.0,  # Shorter segments for subtitles
        merge_short_segments=True
    )
    return TranscriptFormatter(config)


def create_meeting_formatter() -> TranscriptFormatter:
    """Create a formatter optimized for meeting transcripts"""
    config = FormattingConfig(
        output_format=OutputFormat.TEXT,
        include_timestamps=True,
        include_speakers=True,
        timestamp_format=TimestampFormat.HOURS_MINUTES_SECONDS,
        line_break_on_speaker_change=True,
        include_metadata=True,
        speaker_format="{label}: ",
        max_line_length=100
    )
    return TranscriptFormatter(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcript Formatter Test")
    parser.add_argument("--format", type=str, default="text", 
                       choices=[f.value for f in OutputFormat], help="Output format")
    parser.add_argument("--input", type=str, help="Input JSON file with segments")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--include-timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--include-speakers", action="store_true", help="Include speakers")
    parser.add_argument("--include-confidence", action="store_true", help="Include confidence scores")
    args = parser.parse_args()
    
    # Create formatter
    config = FormattingConfig(
        output_format=OutputFormat(args.format),
        include_timestamps=args.include_timestamps,
        include_speakers=args.include_speakers,
        include_confidence=args.include_confidence
    )
    
    formatter = TranscriptFormatter(config)
    
    try:
        # Initialize formatter
        if not formatter.initialize():
            print("Failed to initialize formatter")
            sys.exit(1)
            
        print(f"Formatter status: {formatter.get_status()}")
        
        if args.input:
            # Load segments from JSON file
            with open(args.input, 'r') as f:
                data = json.load(f)
                
            # Create segments
            segments = []
            for seg_data in data.get("segments", []):
                segment = TranscriptSegment(
                    text=seg_data.get("text", ""),
                    speaker_id=seg_data.get("speaker_id", 1),
                    speaker_label=seg_data.get("speaker_label", "Speaker 1"),
                    start_time=seg_data.get("start_time", 0.0),
                    end_time=seg_data.get("end_time", 0.0),
                    duration=seg_data.get("duration", 0.0),
                    confidence=seg_data.get("confidence", 1.0),
                    voice_activity=seg_data.get("voice_activity", 1.0)
                )
                segments.append(segment)
                
            # Format transcript
            formatted_transcript = formatter.format_transcript(segments)
            
            print(f"Formatted {len(segments)} segments")
            print(f"Statistics: {formatted_transcript.statistics}")
            
            if args.output:
                # Export to file
                if formatter.export_to_file(formatted_transcript, args.output):
                    print(f"Exported to {args.output}")
                else:
                    print("Export failed")
            else:
                # Print to console
                print("\nFormatted transcript:")
                print(formatted_transcript.content)
                
        else:
            # Create sample segments
            sample_segments = [
                TranscriptSegment(
                    text="Hello everyone, welcome to today's meeting.",
                    speaker_id=1,
                    speaker_label="Speaker 1",
                    start_time=0.0,
                    end_time=3.0,
                    duration=3.0,
                    confidence=0.95,
                    voice_activity=0.8
                ),
                TranscriptSegment(
                    text="Thank you for having me. I'm excited to discuss the project.",
                    speaker_id=2,
                    speaker_label="Speaker 2",
                    start_time=3.5,
                    end_time=7.0,
                    duration=3.5,
                    confidence=0.92,
                    voice_activity=0.9
                )
            ]
            
            # Format sample
            formatted_transcript = formatter.format_transcript(sample_segments)
            
            print("\nSample formatted transcript:")
            print(formatted_transcript.content)
            
        # Show statistics
        print(f"\nFormatter statistics: {formatter.get_stats()}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        formatter.shutdown()