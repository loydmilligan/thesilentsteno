#!/usr/bin/env python3

"""
Audio Format Converter for The Silent Steno

This module provides real-time audio format conversion capabilities
for the audio pipeline. It handles conversion between different sample
rates, bit depths, channel configurations, and audio codecs to ensure
compatibility between Bluetooth sources and sinks.

Key features:
- Real-time sample rate conversion
- Bit depth conversion (16-bit, 24-bit, 32-bit)
- Channel configuration changes (mono/stereo)
- Audio codec conversion support
- Optimized conversion algorithms
- Format validation and error handling
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleRate(Enum):
    """Supported sample rates"""
    RATE_8000 = 8000
    RATE_11025 = 11025
    RATE_16000 = 16000
    RATE_22050 = 22050
    RATE_44100 = 44100
    RATE_48000 = 48000
    RATE_88200 = 88200
    RATE_96000 = 96000


class BitDepth(Enum):
    """Supported bit depths"""
    BIT_8 = 8
    BIT_16 = 16
    BIT_24 = 24
    BIT_32 = 32


class ChannelConfig(Enum):
    """Channel configurations"""
    MONO = 1
    STEREO = 2
    SURROUND_5_1 = 6
    SURROUND_7_1 = 8


@dataclass
class AudioFormat:
    """Audio format specification"""
    sample_rate: SampleRate
    bit_depth: BitDepth
    channels: ChannelConfig
    codec: str = "PCM"
    endianness: str = "little"  # "little" or "big"


@dataclass
class ConversionSpec:
    """Audio format conversion specification"""
    input_format: AudioFormat
    output_format: AudioFormat
    quality: str = "medium"  # "low", "medium", "high"
    buffer_size: int = 512


class FormatConverter:
    """
    Real-time Audio Format Converter for The Silent Steno
    
    Provides high-quality audio format conversion with minimal latency
    for real-time audio processing pipelines.
    """
    
    def __init__(self):
        """Initialize format converter"""
        self.active_conversions: Dict[str, ConversionSpec] = {}
        self.conversion_cache: Dict[str, Any] = {}
        self.performance_stats = {
            "conversions_performed": 0,
            "total_conversion_time": 0.0,
            "avg_conversion_time": 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Format converter initialized")
    
    def get_supported_formats(self) -> Dict[str, List[Any]]:
        """
        Get list of supported audio formats
        
        Returns:
            Dict containing supported sample rates, bit depths, and channels
        """
        return {
            "sample_rates": [rate.value for rate in SampleRate],
            "bit_depths": [depth.value for depth in BitDepth],
            "channels": [config.value for config in ChannelConfig],
            "codecs": ["PCM", "SBC", "AAC", "aptX"]
        }
    
    def validate_format(self, audio_format: AudioFormat) -> bool:
        """
        Validate audio format specification
        
        Args:
            audio_format: Audio format to validate
            
        Returns:
            bool: True if format is supported
        """
        try:
            supported = self.get_supported_formats()
            
            if audio_format.sample_rate.value not in supported["sample_rates"]:
                return False
            
            if audio_format.bit_depth.value not in supported["bit_depths"]:
                return False
            
            if audio_format.channels.value not in supported["channels"]:
                return False
            
            if audio_format.codec not in supported["codecs"]:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating format: {e}")
            return False
    
    def convert_audio(self, audio_data: np.ndarray, conversion_spec: ConversionSpec) -> Optional[np.ndarray]:
        """
        Convert audio data from one format to another
        
        Args:
            audio_data: Input audio data as numpy array
            conversion_spec: Conversion specification
            
        Returns:
            Converted audio data, or None if conversion failed
        """
        try:
            start_time = time.time()
            
            with self.lock:
                # Validate input format
                if not self.validate_format(conversion_spec.input_format):
                    logger.error("Invalid input format")
                    return None
                
                if not self.validate_format(conversion_spec.output_format):
                    logger.error("Invalid output format")
                    return None
                
                # Start with input data
                converted_data = audio_data.copy()
                
                # Apply conversions in sequence
                converted_data = self._convert_sample_rate(converted_data, conversion_spec)
                converted_data = self._convert_bit_depth(converted_data, conversion_spec)
                converted_data = self._convert_channels(converted_data, conversion_spec)
                
                # Update performance statistics
                conversion_time = time.time() - start_time
                self.performance_stats["conversions_performed"] += 1
                self.performance_stats["total_conversion_time"] += conversion_time
                self.performance_stats["avg_conversion_time"] = (
                    self.performance_stats["total_conversion_time"] / 
                    self.performance_stats["conversions_performed"]
                )
                
                logger.debug(f"Audio conversion completed in {conversion_time*1000:.2f}ms")
                
                return converted_data
        
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return None
    
    def _convert_sample_rate(self, audio_data: np.ndarray, spec: ConversionSpec) -> np.ndarray:
        """Convert audio sample rate"""
        input_rate = spec.input_format.sample_rate.value
        output_rate = spec.output_format.sample_rate.value
        
        if input_rate == output_rate:
            return audio_data
        
        try:
            # Calculate resampling ratio
            ratio = output_rate / input_rate
            
            # Simple linear interpolation resampling
            # In production, would use higher quality algorithms like scipy.signal.resample
            if len(audio_data.shape) == 1:
                # Mono audio
                output_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, output_length)
                resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            else:
                # Multi-channel audio
                output_length = int(audio_data.shape[0] * ratio)
                indices = np.linspace(0, audio_data.shape[0] - 1, output_length)
                resampled = np.zeros((output_length, audio_data.shape[1]))
                
                for channel in range(audio_data.shape[1]):
                    resampled[:, channel] = np.interp(
                        indices, 
                        np.arange(audio_data.shape[0]), 
                        audio_data[:, channel]
                    )
            
            logger.debug(f"Sample rate converted: {input_rate}Hz → {output_rate}Hz")
            return resampled
        
        except Exception as e:
            logger.error(f"Error in sample rate conversion: {e}")
            return audio_data
    
    def _convert_bit_depth(self, audio_data: np.ndarray, spec: ConversionSpec) -> np.ndarray:
        """Convert audio bit depth"""
        input_depth = spec.input_format.bit_depth.value
        output_depth = spec.output_format.bit_depth.value
        
        if input_depth == output_depth:
            return audio_data
        
        try:
            # Convert to float64 for processing
            if audio_data.dtype != np.float64:
                if input_depth == 16:
                    audio_data = audio_data.astype(np.float64) / 32768.0
                elif input_depth == 24:
                    audio_data = audio_data.astype(np.float64) / 8388608.0
                elif input_depth == 32:
                    audio_data = audio_data.astype(np.float64) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float64)
            
            # Convert to target bit depth
            if output_depth == 16:
                converted = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            elif output_depth == 24:
                converted = np.clip(audio_data * 8388607.0, -8388608, 8388607).astype(np.int32)
            elif output_depth == 32:
                converted = np.clip(audio_data * 2147483647.0, -2147483648, 2147483647).astype(np.int32)
            else:
                converted = audio_data.astype(np.float32)
            
            logger.debug(f"Bit depth converted: {input_depth}-bit → {output_depth}-bit")
            return converted
        
        except Exception as e:
            logger.error(f"Error in bit depth conversion: {e}")
            return audio_data
    
    def _convert_channels(self, audio_data: np.ndarray, spec: ConversionSpec) -> np.ndarray:
        """Convert audio channel configuration"""
        input_channels = spec.input_format.channels.value
        output_channels = spec.output_format.channels.value
        
        if input_channels == output_channels:
            return audio_data
        
        try:
            if input_channels == 1 and output_channels == 2:
                # Mono to stereo - duplicate channel
                if len(audio_data.shape) == 1:
                    converted = np.column_stack([audio_data, audio_data])
                else:
                    converted = np.column_stack([audio_data[:, 0], audio_data[:, 0]])
            
            elif input_channels == 2 and output_channels == 1:
                # Stereo to mono - average channels
                if len(audio_data.shape) == 2 and audio_data.shape[1] >= 2:
                    converted = np.mean(audio_data[:, :2], axis=1)
                else:
                    converted = audio_data
            
            else:
                # For other conversions, use simple channel mapping
                if output_channels > input_channels:
                    # Upmix - duplicate existing channels
                    if len(audio_data.shape) == 1:
                        converted = np.tile(audio_data.reshape(-1, 1), (1, output_channels))
                    else:
                        channels_to_add = output_channels - audio_data.shape[1]
                        additional_channels = np.tile(
                            audio_data[:, -1:], (1, channels_to_add)
                        )
                        converted = np.column_stack([audio_data, additional_channels])
                else:
                    # Downmix - take first N channels
                    if len(audio_data.shape) == 2:
                        converted = audio_data[:, :output_channels]
                    else:
                        converted = audio_data
            
            logger.debug(f"Channel configuration converted: {input_channels} → {output_channels}")
            return converted
        
        except Exception as e:
            logger.error(f"Error in channel conversion: {e}")
            return audio_data
    
    def codec_conversion(self, audio_data: np.ndarray, input_codec: str, output_codec: str) -> Optional[np.ndarray]:
        """
        Convert between audio codecs
        
        Args:
            audio_data: Input audio data
            input_codec: Source codec ("PCM", "SBC", "AAC", "aptX")
            output_codec: Target codec
            
        Returns:
            Converted audio data, or None if conversion failed
        """
        try:
            if input_codec == output_codec:
                return audio_data
            
            # For now, assume all codecs can convert through PCM
            # In production, would implement proper codec libraries
            
            if input_codec != "PCM":
                # Decode from source codec to PCM
                decoded_data = self._decode_codec(audio_data, input_codec)
                if decoded_data is None:
                    return None
            else:
                decoded_data = audio_data
            
            if output_codec != "PCM":
                # Encode from PCM to target codec
                encoded_data = self._encode_codec(decoded_data, output_codec)
                return encoded_data
            else:
                return decoded_data
        
        except Exception as e:
            logger.error(f"Error in codec conversion: {e}")
            return None
    
    def _decode_codec(self, audio_data: np.ndarray, codec: str) -> Optional[np.ndarray]:
        """Decode audio from specified codec to PCM"""
        try:
            # Placeholder for codec decoding
            # In production, would use appropriate codec libraries
            logger.debug(f"Decoding from {codec} to PCM")
            return audio_data  # Pass-through for now
        
        except Exception as e:
            logger.error(f"Error decoding {codec}: {e}")
            return None
    
    def _encode_codec(self, audio_data: np.ndarray, codec: str) -> Optional[np.ndarray]:
        """Encode PCM audio to specified codec"""
        try:
            # Placeholder for codec encoding
            # In production, would use appropriate codec libraries
            logger.debug(f"Encoding PCM to {codec}")
            return audio_data  # Pass-through for now
        
        except Exception as e:
            logger.error(f"Error encoding to {codec}: {e}")
            return None
    
    def create_conversion_spec(self, 
                              input_sample_rate: int,
                              input_bit_depth: int,
                              input_channels: int,
                              output_sample_rate: int,
                              output_bit_depth: int,
                              output_channels: int,
                              input_codec: str = "PCM",
                              output_codec: str = "PCM") -> Optional[ConversionSpec]:
        """
        Create a conversion specification from parameters
        
        Returns:
            ConversionSpec object, or None if invalid parameters
        """
        try:
            input_format = AudioFormat(
                sample_rate=SampleRate(input_sample_rate),
                bit_depth=BitDepth(input_bit_depth),
                channels=ChannelConfig(input_channels),
                codec=input_codec
            )
            
            output_format = AudioFormat(
                sample_rate=SampleRate(output_sample_rate),
                bit_depth=BitDepth(output_bit_depth),
                channels=ChannelConfig(output_channels),
                codec=output_codec
            )
            
            return ConversionSpec(
                input_format=input_format,
                output_format=output_format
            )
        
        except (ValueError, Exception) as e:
            logger.error(f"Error creating conversion spec: {e}")
            return None
    
    def get_conversion_performance(self) -> Dict[str, Any]:
        """
        Get format conversion performance statistics
        
        Returns:
            Dict containing performance metrics
        """
        with self.lock:
            return {
                "conversions_performed": self.performance_stats["conversions_performed"],
                "total_conversion_time_ms": self.performance_stats["total_conversion_time"] * 1000,
                "avg_conversion_time_ms": self.performance_stats["avg_conversion_time"] * 1000,
                "active_conversions": len(self.active_conversions),
                "cache_size": len(self.conversion_cache)
            }


if __name__ == "__main__":
    # Basic test when run directly
    print("Format Converter Test")
    print("=" * 50)
    
    converter = FormatConverter()
    
    print("Supported formats:")
    formats = converter.get_supported_formats()
    for format_type, values in formats.items():
        print(f"  {format_type}: {values}")
    
    # Test format creation
    spec = converter.create_conversion_spec(
        input_sample_rate=44100,
        input_bit_depth=16,
        input_channels=2,
        output_sample_rate=48000,
        output_bit_depth=24,
        output_channels=2
    )
    
    if spec:
        print(f"\nConversion spec created: {spec.input_format.sample_rate.value}Hz/{spec.input_format.bit_depth.value}bit → {spec.output_format.sample_rate.value}Hz/{spec.output_format.bit_depth.value}bit")
        
        # Test with dummy audio data
        dummy_audio = np.random.rand(1024, 2).astype(np.float32)
        print(f"Input audio shape: {dummy_audio.shape}")
        
        converted = converter.convert_audio(dummy_audio, spec)
        if converted is not None:
            print(f"Output audio shape: {converted.shape}")
            print("Conversion successful!")
        else:
            print("Conversion failed")
    
    # Performance stats
    stats = converter.get_conversion_performance()
    print(f"\nPerformance: {stats}")
