#!/usr/bin/env python3

"""
Settings Manager for The Silent Steno

Handles all application settings with persistence, validation, and defaults.
Provides a centralized system for managing user preferences across
audio, transcription, data, and interface settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class AudioSettings:
    """Audio and recording related settings"""
    microphone_source: str = "default"
    input_gain: float = 0.5  # 0.0 to 1.0
    noise_suppression: str = "low"  # off, low, high
    voice_activity_detection: bool = True
    vad_sensitivity: float = 0.5  # 0.0 to 1.0
    recording_format: str = "wav"  # wav, mp3
    recording_quality: str = "high"  # low, medium, high

@dataclass
class TranscriptionSettings:
    """Transcription and AI related settings"""
    transcription_language: str = "en"
    whisper_model: str = "base"  # base, small, medium, large
    custom_vocabulary: str = ""  # comma-separated terms
    speaker_diarization: str = "manual"  # manual, automatic
    automatic_punctuation: bool = True
    profanity_filter: bool = False
    post_meeting_summaries: bool = True
    use_gemini_enhancement: bool = False
    gemini_api_key: str = ""

@dataclass
class DataSettings:
    """Data handling and export settings"""
    default_save_location: str = "demo_sessions"
    cloud_sync_enabled: bool = False
    cloud_provider: str = "none"  # none, google_drive, dropbox
    auto_delete_policy: str = "never"  # never, 30_days, 90_days
    default_export_format: str = "txt"  # txt, md, docx
    include_timestamps: bool = True
    include_speaker_labels: bool = True
    include_highlight_markers: bool = True

@dataclass
class InterfaceSettings:
    """UI and interface related settings"""
    theme: str = "dark"  # dark, light
    font_size: str = "medium"  # small, medium, large
    button_size: str = "large"  # medium, large, extra_large
    screen_orientation_lock: bool = False
    show_waveform: bool = True
    auto_scroll_transcript: bool = True

class SettingsManager:
    """
    Centralized settings management system
    
    Handles loading, saving, validation, and providing defaults for all
    application settings across different categories.
    """
    
    def __init__(self, settings_file: str = "config/settings.json"):
        """
        Initialize settings manager
        
        Args:
            settings_file: Path to settings file relative to project root
        """
        self.settings_file = Path(settings_file)
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with defaults
        self.audio = AudioSettings()
        self.transcription = TranscriptionSettings()
        self.data = DataSettings()
        self.interface = InterfaceSettings()
        
        # Load existing settings if available
        self.load_settings()
        
        logger.info("Settings manager initialized")
    
    def load_settings(self) -> bool:
        """
        Load settings from file
        
        Returns:
            True if settings loaded successfully, False otherwise
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                
                # Load each category with validation
                if 'audio' in data:
                    self.audio = AudioSettings(**{k: v for k, v in data['audio'].items() 
                                                if k in AudioSettings.__dataclass_fields__})
                
                if 'transcription' in data:
                    self.transcription = TranscriptionSettings(**{k: v for k, v in data['transcription'].items() 
                                                               if k in TranscriptionSettings.__dataclass_fields__})
                
                if 'data' in data:
                    self.data = DataSettings(**{k: v for k, v in data['data'].items() 
                                              if k in DataSettings.__dataclass_fields__})
                
                if 'interface' in data:
                    self.interface = InterfaceSettings(**{k: v for k, v in data['interface'].items() 
                                                        if k in InterfaceSettings.__dataclass_fields__})
                
                logger.info(f"Settings loaded from {self.settings_file}")
                return True
            else:
                logger.info("No settings file found, using defaults")
                return False
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """
        Save current settings to file
        
        Returns:
            True if settings saved successfully, False otherwise
        """
        try:
            settings_data = {
                'audio': asdict(self.audio),
                'transcription': asdict(self.transcription),
                'data': asdict(self.data),
                'interface': asdict(self.interface)
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)
            
            logger.info(f"Settings saved to {self.settings_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            'audio': asdict(self.audio),
            'transcription': asdict(self.transcription),
            'data': asdict(self.data),
            'interface': asdict(self.interface)
        }
    
    def update_settings(self, category: str, updates: Dict[str, Any]) -> bool:
        """
        Update settings for a specific category
        
        Args:
            category: Settings category (audio, transcription, data, interface)
            updates: Dictionary of setting key-value pairs to update
            
        Returns:
            True if updates were successful, False otherwise
        """
        try:
            if category == 'audio':
                for key, value in updates.items():
                    if hasattr(self.audio, key):
                        setattr(self.audio, key, value)
                        
            elif category == 'transcription':
                for key, value in updates.items():
                    if hasattr(self.transcription, key):
                        setattr(self.transcription, key, value)
                        
            elif category == 'data':
                for key, value in updates.items():
                    if hasattr(self.data, key):
                        setattr(self.data, key, value)
                        
            elif category == 'interface':
                for key, value in updates.items():
                    if hasattr(self.interface, key):
                        setattr(self.interface, key, value)
            else:
                logger.warning(f"Unknown settings category: {category}")
                return False
            
            # Save after updating
            return self.save_settings()
            
        except Exception as e:
            logger.error(f"Error updating {category} settings: {e}")
            return False
    
    def reset_to_defaults(self, category: Optional[str] = None) -> bool:
        """
        Reset settings to defaults
        
        Args:
            category: Specific category to reset, or None for all categories
            
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            if category is None or category == 'audio':
                self.audio = AudioSettings()
            if category is None or category == 'transcription':
                self.transcription = TranscriptionSettings()
            if category is None or category == 'data':
                self.data = DataSettings()
            if category is None or category == 'interface':
                self.interface = InterfaceSettings()
            
            return self.save_settings()
            
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return False
    
    def validate_settings(self) -> Dict[str, list]:
        """
        Validate current settings and return any issues
        
        Returns:
            Dictionary of validation errors by category
        """
        errors = {
            'audio': [],
            'transcription': [],
            'data': [],
            'interface': []
        }
        
        # Audio validation
        if not 0.0 <= self.audio.input_gain <= 1.0:
            errors['audio'].append("Input gain must be between 0.0 and 1.0")
        
        if self.audio.noise_suppression not in ['off', 'low', 'high']:
            errors['audio'].append("Noise suppression must be 'off', 'low', or 'high'")
        
        # Transcription validation
        if self.transcription.whisper_model not in ['base', 'small', 'medium', 'large']:
            errors['transcription'].append("Invalid Whisper model selection")
            
        if self.transcription.use_gemini_enhancement and not self.transcription.gemini_api_key.strip():
            errors['transcription'].append("Gemini API key required when enhancement is enabled")
        
        # Data validation
        if not os.path.exists(self.data.default_save_location):
            try:
                os.makedirs(self.data.default_save_location, exist_ok=True)
            except:
                errors['data'].append("Cannot create default save location")
        
        # Interface validation
        if self.interface.theme not in ['dark', 'light']:
            errors['interface'].append("Invalid theme selection")
        
        return {k: v for k, v in errors.items() if v}  # Return only categories with errors

# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

def get_setting(category: str, key: str, default: Any = None) -> Any:
    """
    Convenience function to get a specific setting
    
    Args:
        category: Settings category
        key: Setting key
        default: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    manager = get_settings_manager()
    settings_obj = getattr(manager, category, None)
    if settings_obj is None:
        return default
    return getattr(settings_obj, key, default)

def set_setting(category: str, key: str, value: Any) -> bool:
    """
    Convenience function to set a specific setting
    
    Args:
        category: Settings category
        key: Setting key
        value: New value
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_settings_manager()
    return manager.update_settings(category, {key: value})

if __name__ == "__main__":
    # Basic test when run directly
    print("Settings Manager Test")
    print("=" * 50)
    
    # Create settings manager
    manager = SettingsManager("test_settings.json")
    
    # Display current settings
    print("Current settings:")
    print(json.dumps(manager.get_all_settings(), indent=2))
    
    # Test updating a setting
    print("\nUpdating transcription language to 'es'...")
    manager.update_settings('transcription', {'transcription_language': 'es'})
    
    # Test validation
    print("\nValidation results:")
    errors = manager.validate_settings()
    if errors:
        print(json.dumps(errors, indent=2))
    else:
        print("All settings valid!")
    
    print("\nTest complete")