#!/usr/bin/env python3

"""
High-Quality Audio Recorder for The Silent Steno

This module provides the core audio recording functionality with support for
multiple high-quality formats (FLAC/WAV/MP3) and real-time processing. It
integrates with the audio pipeline to capture live audio while maintaining
the low-latency forwarding capability.

Key features:
- High-quality audio recording (FLAC/WAV/MP3)
- Real-time recording from audio pipeline
- Multiple quality presets and format options
- Thread-safe recording operations
- Real-time audio level monitoring
- File format conversion and optimization
- Integration with audio preprocessing
"""

import os
import wave
import threading
import time
import logging
import subprocess
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available, limited format support")

try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, limited conversion support")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecordingFormat(Enum):
    """Supported recording formats"""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    OGG = "ogg"


class QualityPreset(Enum):
    """Recording quality presets"""
    LOW_LATENCY = "low_latency"      # 44.1kHz/16-bit, optimized for speed
    BALANCED = "balanced"            # 44.1kHz/16-bit, good quality/performance
    HIGH_QUALITY = "high_quality"    # 48kHz/24-bit, best quality
    ARCHIVAL = "archival"           # 48kHz/24-bit FLAC, maximum quality


class RecordingState(Enum):
    """Recording states"""
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RecordingConfig:
    """Recording configuration parameters"""
    format: RecordingFormat = RecordingFormat.FLAC
    quality_preset: QualityPreset = QualityPreset.BALANCED
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    buffer_size: int = 1024
    enable_preprocessing: bool = True
    enable_level_monitoring: bool = True
    compression_level: int = 5  # For FLAC (0-8)
    mp3_bitrate: int = 320  # For MP3


@dataclass
class RecordingInfo:
    """Information about a recording"""
    session_id: str
    file_path: str
    format: RecordingFormat
    sample_rate: int
    channels: int
    bit_depth: int
    duration_seconds: float
    file_size_bytes: int
    peak_level_db: float
    rms_level_db: float
    quality_score: float
    metadata: Dict[str, Any]
    created_at: float
    completed_at: Optional[float] = None


class AudioRecorder:
    """
    High-Quality Audio Recorder for The Silent Steno
    
    Provides comprehensive audio recording capabilities with multiple format
    support and real-time processing integration.
    """
    
    def __init__(self, storage_root: str = "recordings"):
        """Initialize audio recorder"""
        self.storage_root = storage_root
        
        # Recording state
        self.active_recordings: Dict[str, Dict[str, Any]] = {}
        self.recording_lock = threading.RLock()
        
        # Audio processing
        self.audio_pipeline = None
        self.preprocessor = None
        self.level_monitor = None
        
        # Performance tracking
        self.performance_stats = {
            "recordings_started": 0,
            "recordings_completed": 0,
            "total_recording_time": 0.0,
            "total_data_written": 0,
            "average_quality_score": 0.0
        }
        
        # Callbacks
        self.recording_callbacks: List[Callable] = []
        self.level_callbacks: List[Callable] = []
        
        # Initialize storage
        os.makedirs(storage_root, exist_ok=True)
        
        logger.info(f"Audio recorder initialized with storage: {storage_root}")
    
    def set_audio_pipeline(self, pipeline) -> None:
        """Set audio pipeline for live recording"""
        self.audio_pipeline = pipeline
        if pipeline:
            # Register for audio data callbacks
            pipeline.add_audio_callback(self._on_audio_data)
    
    def set_preprocessor(self, preprocessor) -> None:
        """Set audio preprocessor"""
        self.preprocessor = preprocessor
    
    def set_level_monitor(self, monitor) -> None:
        """Set level monitor"""
        self.level_monitor = monitor
    
    def add_recording_callback(self, callback: Callable[[str, RecordingInfo], None]) -> None:
        """Add callback for recording events"""
        self.recording_callbacks.append(callback)
    
    def add_level_callback(self, callback: Callable[[str, Dict[str, float]], None]) -> None:
        """Add callback for level updates"""
        self.level_callbacks.append(callback)
    
    def _notify_recording_event(self, session_id: str, recording_info: RecordingInfo) -> None:
        """Notify callbacks of recording events"""
        for callback in self.recording_callbacks:
            try:
                callback(session_id, recording_info)
            except Exception as e:
                logger.error(f"Error in recording callback: {e}")
    
    def _notify_level_update(self, session_id: str, levels: Dict[str, float]) -> None:
        """Notify callbacks of level updates"""
        for callback in self.level_callbacks:
            try:
                callback(session_id, levels)
            except Exception as e:
                logger.error(f"Error in level callback: {e}")
    
    def start_recording(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Start recording for a session
        
        Args:
            session_id: Unique session identifier
            config: Recording configuration
            
        Returns:
            True if recording started successfully
        """
        try:
            with self.recording_lock:
                if session_id in self.active_recordings:
                    logger.warning(f"Recording already active for session {session_id}")
                    return False
                
                # Parse configuration
                recording_config = self._parse_recording_config(config)
                file_path = config.get('file_path') or self._generate_file_path(session_id, recording_config)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Initialize recording context
                recording_context = {
                    'session_id': session_id,
                    'config': recording_config,
                    'file_path': file_path,
                    'state': RecordingState.STARTING,
                    'start_time': time.time(),
                    'audio_data': [],
                    'audio_writer': None,
                    'temp_file_path': None,
                    'total_frames': 0,
                    'peak_level': -60.0,
                    'rms_levels': [],
                    'quality_metrics': []
                }
                
                # Create audio writer
                audio_writer = self._create_audio_writer(file_path, recording_config)
                if not audio_writer:
                    logger.error(f"Failed to create audio writer for {file_path}")
                    return False
                
                recording_context['audio_writer'] = audio_writer
                
                # Store recording context
                self.active_recordings[session_id] = recording_context
                
                # Update state
                recording_context['state'] = RecordingState.RECORDING
                
                # Update statistics
                self.performance_stats['recordings_started'] += 1
                
                logger.info(f"Recording started for session {session_id}: {file_path}")
                return True
        
        except Exception as e:
            logger.error(f"Error starting recording for session {session_id}: {e}")
            return False
    
    def stop_recording(self, session_id: str) -> Optional[RecordingInfo]:
        """
        Stop recording for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            RecordingInfo if successful, None otherwise
        """
        try:
            with self.recording_lock:
                recording_context = self.active_recordings.get(session_id)
                if not recording_context:
                    logger.warning(f"No active recording for session {session_id}")
                    return None
                
                if recording_context['state'] not in [RecordingState.RECORDING, RecordingState.PAUSED]:
                    logger.warning(f"Recording for session {session_id} is not active")
                    return None
                
                # Update state
                recording_context['state'] = RecordingState.STOPPING
                
                # Finalize audio file
                recording_info = self._finalize_recording(recording_context)
                
                # Cleanup
                del self.active_recordings[session_id]
                
                # Update statistics
                self.performance_stats['recordings_completed'] += 1
                if recording_info:
                    self.performance_stats['total_recording_time'] += recording_info.duration_seconds
                    self.performance_stats['total_data_written'] += recording_info.file_size_bytes
                
                logger.info(f"Recording stopped for session {session_id}")
                
                # Notify callbacks
                if recording_info:
                    self._notify_recording_event(session_id, recording_info)
                
                return recording_info
        
        except Exception as e:
            logger.error(f"Error stopping recording for session {session_id}: {e}")
            return None
    
    def pause_recording(self, session_id: str) -> bool:
        """
        Pause recording for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            with self.recording_lock:
                recording_context = self.active_recordings.get(session_id)
                if not recording_context or recording_context['state'] != RecordingState.RECORDING:
                    return False
                
                recording_context['state'] = RecordingState.PAUSED
                recording_context['pause_time'] = time.time()
                
                logger.info(f"Recording paused for session {session_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error pausing recording for session {session_id}: {e}")
            return False
    
    def resume_recording(self, session_id: str) -> bool:
        """
        Resume recording for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            with self.recording_lock:
                recording_context = self.active_recordings.get(session_id)
                if not recording_context or recording_context['state'] != RecordingState.PAUSED:
                    return False
                
                recording_context['state'] = RecordingState.RECORDING
                
                # Adjust timing for pause duration
                if 'pause_time' in recording_context:
                    pause_duration = time.time() - recording_context['pause_time']
                    recording_context['start_time'] += pause_duration
                    del recording_context['pause_time']
                
                logger.info(f"Recording resumed for session {session_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error resuming recording for session {session_id}: {e}")
            return False
    
    def _on_audio_data(self, audio_data: np.ndarray) -> None:
        """Handle audio data from pipeline"""
        try:
            with self.recording_lock:
                for session_id, recording_context in self.active_recordings.items():
                    if recording_context['state'] == RecordingState.RECORDING:
                        self._process_audio_data(session_id, recording_context, audio_data)
        
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def _process_audio_data(self, session_id: str, recording_context: Dict[str, Any], 
                          audio_data: np.ndarray) -> None:
        """Process audio data for recording"""
        try:
            config = recording_context['config']
            
            # Apply preprocessing if enabled
            processed_data = audio_data
            if config.enable_preprocessing and self.preprocessor:
                processed_data = self.preprocessor.process_audio(audio_data)
            
            # Monitor audio levels
            if config.enable_level_monitoring:
                levels = self._calculate_audio_levels(processed_data)
                recording_context['peak_level'] = max(recording_context['peak_level'], levels['peak_db'])
                recording_context['rms_levels'].append(levels['rms_db'])
                
                # Notify level callbacks
                self._notify_level_update(session_id, levels)
            
            # Write audio data
            audio_writer = recording_context['audio_writer']
            if audio_writer:
                self._write_audio_data(audio_writer, processed_data, config)
                recording_context['total_frames'] += len(processed_data)
        
        except Exception as e:
            logger.error(f"Error processing audio data for session {session_id}: {e}")
    
    def _calculate_audio_levels(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate audio levels from data"""
        try:
            if len(audio_data) == 0:
                return {'peak_db': -60.0, 'rms_db': -60.0}
            
            # Peak level
            peak = np.max(np.abs(audio_data))
            peak_db = 20 * np.log10(peak) if peak > 0 else -60.0
            
            # RMS level
            rms = np.sqrt(np.mean(audio_data ** 2))
            rms_db = 20 * np.log10(rms) if rms > 0 else -60.0
            
            return {'peak_db': peak_db, 'rms_db': rms_db}
        
        except Exception:
            return {'peak_db': -60.0, 'rms_db': -60.0}
    
    def _parse_recording_config(self, config: Dict[str, Any]) -> RecordingConfig:
        """Parse recording configuration from dictionary"""
        return RecordingConfig(
            format=RecordingFormat(config.get('format', 'flac')),
            quality_preset=QualityPreset(config.get('quality', 'balanced')),
            sample_rate=config.get('sample_rate', 44100),
            channels=config.get('channels', 2),
            bit_depth=config.get('bit_depth', 16),
            buffer_size=config.get('buffer_size', 1024),
            enable_preprocessing=config.get('preprocessing', True),
            enable_level_monitoring=config.get('level_monitoring', True),
            compression_level=config.get('compression_level', 5),
            mp3_bitrate=config.get('mp3_bitrate', 320)
        )
    
    def _generate_file_path(self, session_id: str, config: RecordingConfig) -> str:
        """Generate file path for recording"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}_{session_id[:8]}.{config.format.value}"
        return os.path.join(self.storage_root, "sessions", filename)
    
    def _create_audio_writer(self, file_path: str, config: RecordingConfig):
        """Create appropriate audio writer for format"""
        try:
            if config.format == RecordingFormat.WAV:
                return self._create_wav_writer(file_path, config)
            elif config.format == RecordingFormat.FLAC and SOUNDFILE_AVAILABLE:
                return self._create_flac_writer(file_path, config)
            elif config.format == RecordingFormat.MP3:
                return self._create_mp3_writer(file_path, config)
            else:
                # Fallback to WAV
                logger.warning(f"Format {config.format.value} not supported, using WAV")
                fallback_path = file_path.replace(f".{config.format.value}", ".wav")
                return self._create_wav_writer(fallback_path, config)
        
        except Exception as e:
            logger.error(f"Error creating audio writer: {e}")
            return None
    
    def _create_wav_writer(self, file_path: str, config: RecordingConfig):
        """Create WAV file writer"""
        try:
            writer = wave.open(file_path, 'wb')
            writer.setnchannels(config.channels)
            writer.setsampwidth(config.bit_depth // 8)
            writer.setframerate(config.sample_rate)
            return writer
        except Exception as e:
            logger.error(f"Error creating WAV writer: {e}")
            return None
    
    def _create_flac_writer(self, file_path: str, config: RecordingConfig):
        """Create FLAC file writer using soundfile"""
        try:
            if not SOUNDFILE_AVAILABLE:
                return None
            
            # soundfile writer context
            writer = {
                'file_path': file_path,
                'config': config,
                'data_buffer': []
            }
            return writer
        except Exception as e:
            logger.error(f"Error creating FLAC writer: {e}")
            return None
    
    def _create_mp3_writer(self, file_path: str, config: RecordingConfig):
        """Create MP3 writer (using ffmpeg)"""
        try:
            # Create temporary WAV file first
            temp_wav_path = file_path.replace('.mp3', '_temp.wav')
            wav_writer = self._create_wav_writer(temp_wav_path, config)
            
            if wav_writer:
                return {
                    'wav_writer': wav_writer,
                    'temp_wav_path': temp_wav_path,
                    'final_mp3_path': file_path,
                    'config': config
                }
        except Exception as e:
            logger.error(f"Error creating MP3 writer: {e}")
            return None
    
    def _write_audio_data(self, audio_writer, audio_data: np.ndarray, config: RecordingConfig) -> None:
        """Write audio data to file"""
        try:
            if isinstance(audio_writer, wave.Wave_write):
                # WAV writer
                audio_bytes = self._convert_to_bytes(audio_data, config.bit_depth)
                audio_writer.writeframes(audio_bytes)
            
            elif isinstance(audio_writer, dict):
                if 'data_buffer' in audio_writer:
                    # FLAC writer (soundfile)
                    audio_writer['data_buffer'].append(audio_data)
                elif 'wav_writer' in audio_writer:
                    # MP3 writer (WAV intermediate)
                    audio_bytes = self._convert_to_bytes(audio_data, config.bit_depth)
                    audio_writer['wav_writer'].writeframes(audio_bytes)
        
        except Exception as e:
            logger.error(f"Error writing audio data: {e}")
    
    def _convert_to_bytes(self, audio_data: np.ndarray, bit_depth: int) -> bytes:
        """Convert audio data to bytes for writing"""
        try:
            if bit_depth == 16:
                # Convert to 16-bit signed integers
                audio_int = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            elif bit_depth == 24:
                # Convert to 24-bit signed integers
                audio_int = np.clip(audio_data * 8388607, -8388608, 8388607).astype(np.int32)
            elif bit_depth == 32:
                # Convert to 32-bit signed integers
                audio_int = np.clip(audio_data * 2147483647, -2147483648, 2147483647).astype(np.int32)
            else:
                # Default to 16-bit
                audio_int = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            return audio_int.tobytes()
        
        except Exception as e:
            logger.error(f"Error converting audio to bytes: {e}")
            return b''
    
    def _finalize_recording(self, recording_context: Dict[str, Any]) -> Optional[RecordingInfo]:
        """Finalize recording and create recording info"""
        try:
            config = recording_context['config']
            file_path = recording_context['file_path']
            audio_writer = recording_context['audio_writer']
            
            # Close audio writer
            if isinstance(audio_writer, wave.Wave_write):
                audio_writer.close()
            
            elif isinstance(audio_writer, dict):
                if 'data_buffer' in audio_writer and SOUNDFILE_AVAILABLE:
                    # FLAC writer - write buffered data
                    audio_data = np.concatenate(audio_writer['data_buffer'])
                    sf.write(file_path, audio_data, config.sample_rate, 
                           format='FLAC', subtype=f'PCM_{config.bit_depth}')
                
                elif 'wav_writer' in audio_writer:
                    # MP3 writer - convert WAV to MP3
                    audio_writer['wav_writer'].close()
                    self._convert_wav_to_mp3(
                        audio_writer['temp_wav_path'],
                        audio_writer['final_mp3_path'],
                        config.mp3_bitrate
                    )
                    # Clean up temp file
                    try:
                        os.remove(audio_writer['temp_wav_path'])
                    except:
                        pass
            
            # Calculate recording info
            duration = time.time() - recording_context['start_time']
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Calculate quality metrics
            rms_levels = recording_context.get('rms_levels', [])
            avg_rms = np.mean(rms_levels) if rms_levels else -60.0
            quality_score = self._calculate_quality_score(
                recording_context['peak_level'], avg_rms, duration
            )
            
            recording_info = RecordingInfo(
                session_id=recording_context['session_id'],
                file_path=file_path,
                format=config.format,
                sample_rate=config.sample_rate,
                channels=config.channels,
                bit_depth=config.bit_depth,
                duration_seconds=duration,
                file_size_bytes=file_size,
                peak_level_db=recording_context['peak_level'],
                rms_level_db=avg_rms,
                quality_score=quality_score,
                metadata={
                    'total_frames': recording_context['total_frames'],
                    'quality_preset': config.quality_preset.value,
                    'preprocessing_enabled': config.enable_preprocessing
                },
                created_at=recording_context['start_time'],
                completed_at=time.time()
            )
            
            return recording_info
        
        except Exception as e:
            logger.error(f"Error finalizing recording: {e}")
            return None
    
    def _convert_wav_to_mp3(self, wav_path: str, mp3_path: str, bitrate: int) -> bool:
        """Convert WAV to MP3 using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', wav_path,
                '-codec:a', 'libmp3lame',
                '-b:a', f'{bitrate}k',
                '-y', mp3_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"Error converting WAV to MP3: {e}")
            return False
    
    def _calculate_quality_score(self, peak_db: float, avg_rms_db: float, duration: float) -> float:
        """Calculate recording quality score (0-1)"""
        try:
            # Basic quality scoring based on levels and duration
            peak_score = max(0, min(1, (peak_db + 20) / 20))  # -20dB to 0dB = good range
            rms_score = max(0, min(1, (avg_rms_db + 40) / 20))  # -40dB to -20dB = good range
            duration_score = min(1, duration / 60)  # Up to 1 minute for full score
            
            # Weighted average
            quality_score = (peak_score * 0.4 + rms_score * 0.4 + duration_score * 0.2)
            return quality_score
        
        except Exception:
            return 0.5  # Default neutral score
    
    def get_recording_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current recording information
        
        Args:
            session_id: Session identifier
            
        Returns:
            Recording information if active, None otherwise
        """
        with self.recording_lock:
            recording_context = self.active_recordings.get(session_id)
            if not recording_context:
                return None
            
            duration = time.time() - recording_context['start_time']
            return {
                'session_id': session_id,
                'state': recording_context['state'].value,
                'duration_seconds': duration,
                'total_frames': recording_context['total_frames'],
                'peak_level_db': recording_context['peak_level'],
                'file_path': recording_context['file_path'],
                'format': recording_context['config'].format.value
            }
    
    def export_formats(self, file_path: str, target_formats: List[str]) -> Dict[str, str]:
        """
        Export recording to multiple formats
        
        Args:
            file_path: Source file path
            target_formats: List of target formats
            
        Returns:
            Dict mapping format to output file path
        """
        results = {}
        
        try:
            for format_name in target_formats:
                if format_name.lower() == 'mp3':
                    output_path = file_path.replace('.flac', '.mp3').replace('.wav', '.mp3')
                    if self._convert_to_mp3(file_path, output_path):
                        results[format_name] = output_path
                
                elif format_name.lower() == 'wav':
                    output_path = file_path.replace('.flac', '.wav').replace('.mp3', '.wav')
                    if self._convert_to_wav(file_path, output_path):
                        results[format_name] = output_path
                
                elif format_name.lower() == 'flac':
                    output_path = file_path.replace('.wav', '.flac').replace('.mp3', '.flac')
                    if self._convert_to_flac(file_path, output_path):
                        results[format_name] = output_path
        
        except Exception as e:
            logger.error(f"Error exporting formats: {e}")
        
        return results
    
    def _convert_to_mp3(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to MP3"""
        return self._convert_wav_to_mp3(input_path, output_path, 320)
    
    def _convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV"""
        try:
            cmd = ['ffmpeg', '-i', input_path, '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            return False
    
    def _convert_to_flac(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to FLAC"""
        try:
            cmd = ['ffmpeg', '-i', input_path, '-c:a', 'flac', '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error converting to FLAC: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get recording performance statistics"""
        with self.recording_lock:
            active_count = len(self.active_recordings)
            
            # Calculate average quality score
            if self.performance_stats['recordings_completed'] > 0:
                self.performance_stats['average_quality_score'] = (
                    self.performance_stats.get('total_quality_score', 0) / 
                    self.performance_stats['recordings_completed']
                )
            
            return {
                **self.performance_stats,
                'active_recordings': active_count,
                'supported_formats': [fmt.value for fmt in RecordingFormat],
                'soundfile_available': SOUNDFILE_AVAILABLE,
                'pydub_available': PYDUB_AVAILABLE
            }


if __name__ == "__main__":
    # Basic test when run directly
    print("Audio Recorder Test")
    print("=" * 50)
    
    recorder = AudioRecorder("test_recordings")
    
    def on_recording_event(session_id, recording_info):
        print(f"Recording event: {session_id[:8]} - {recording_info.format.value}")
    
    def on_level_update(session_id, levels):
        print(f"Levels: {session_id[:8]} - Peak: {levels['peak_db']:.1f}dB, RMS: {levels['rms_db']:.1f}dB")
    
    recorder.add_recording_callback(on_recording_event)
    recorder.add_level_callback(on_level_update)
    
    # Test recording configuration
    config = {
        'format': 'flac',
        'quality': 'balanced',
        'sample_rate': 44100,
        'channels': 2,
        'preprocessing': True
    }
    
    session_id = "test-session-001"
    
    print("Starting recording...")
    if recorder.start_recording(session_id, config):
        print("Recording started successfully")
        
        # Simulate some audio data
        for i in range(5):
            dummy_audio = np.random.rand(1024, 2) * 0.1
            recorder._on_audio_data(dummy_audio)
            time.sleep(0.1)
        
        # Get recording info
        info = recorder.get_recording_info(session_id)
        if info:
            print(f"Recording info: {info}")
        
        # Stop recording
        print("Stopping recording...")
        recording_info = recorder.stop_recording(session_id)
        if recording_info:
            print(f"Recording completed: {recording_info.file_path}")
            print(f"Duration: {recording_info.duration_seconds:.1f}s")
            print(f"Quality score: {recording_info.quality_score:.2f}")
    
    # Performance stats
    stats = recorder.get_performance_stats()
    print(f"Performance stats: {stats}")