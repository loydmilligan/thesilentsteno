#!/usr/bin/env python3

"""
Transcription Configuration for The Silent Steno

This module provides configuration management for the transcription system,
allowing easy switching between CPU and Hailo backends.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TranscriptionConfig:
    """Configuration manager for transcription settings"""
    
    DEFAULT_CONFIG = {
        "backend": "cpu",  # "cpu" or "hailo"
        "model_name": "base",  # Whisper model name
        "device": "auto",  # Device selection for backend
        "batch_size": 1,  # Batch size for processing
        "language": None,  # Auto-detect if None
        "temperature": 0.0,  # Sampling temperature
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "hailo_config": {
            "model_path": "/opt/hailo/models/whisper_base.hef",
            "runtime_mode": "performance",  # "performance" or "power_save"
            "batch_timeout_ms": 100
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.config_file = config_file or os.path.expanduser("~/.silentsteno/transcription_config.json")
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            self.load()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
                logger.info(f"Loaded transcription config from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_file}: {e}")
        
        return self.config
    
    def save(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved transcription config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(updates)
    
    def get_backend_config(self) -> Dict[str, Any]:
        """Get configuration for current backend"""
        backend = self.config.get("backend", "cpu")
        
        if backend == "hailo":
            return {
                "backend": "hailo",
                "model_name": self.config.get("model_name", "base"),
                **self.config.get("hailo_config", {})
            }
        else:
            return {
                "backend": "cpu",
                "model_name": self.config.get("model_name", "base"),
                "device": self.config.get("device", "auto"),
                "language": self.config.get("language"),
                "temperature": self.config.get("temperature", 0.0)
            }
    
    def switch_backend(self, backend: str) -> bool:
        """Switch to a different backend"""
        if backend not in ["cpu", "hailo"]:
            logger.error(f"Invalid backend: {backend}")
            return False
        
        old_backend = self.config.get("backend")
        self.config["backend"] = backend
        
        # Save the change
        if self.save():
            logger.info(f"Switched transcription backend from {old_backend} to {backend}")
            return True
        
        # Revert on save failure
        self.config["backend"] = old_backend
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self.config.copy()


# Global configuration instance
_global_config = None


def get_transcription_config() -> TranscriptionConfig:
    """Get global transcription configuration"""
    global _global_config
    if _global_config is None:
        _global_config = TranscriptionConfig()
    return _global_config


def set_transcription_backend(backend: str) -> bool:
    """Convenience function to switch backend"""
    config = get_transcription_config()
    return config.switch_backend(backend)


if __name__ == "__main__":
    # Test configuration
    print("Transcription Configuration Test")
    print("=" * 50)
    
    config = TranscriptionConfig()
    print(f"Current backend: {config.get('backend')}")
    print(f"Model name: {config.get('model_name')}")
    print(f"Config file: {config.config_file}")
    
    print("\nBackend-specific config:")
    print(json.dumps(config.get_backend_config(), indent=2))
    
    # Test switching
    print("\nTesting backend switch...")
    if config.switch_backend("hailo"):
        print(f"Switched to: {config.get('backend')}")
        print("Hailo config:", json.dumps(config.get_backend_config(), indent=2))
        
        # Switch back
        config.switch_backend("cpu")
        print(f"Switched back to: {config.get('backend')}")