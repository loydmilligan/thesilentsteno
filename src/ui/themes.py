"""
UI Theming System with Dark/Light Modes

This module provides a comprehensive theming system supporting dark/light modes
with customizable color schemes for The Silent Steno.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
import threading
import time
import logging
import json
import os

try:
    from kivy.event import EventDispatcher
    from kivy.metrics import dp
    from kivy.utils import get_color_from_hex
except ImportError:
    raise ImportError("Kivy not available. Install with: pip install kivy")

logger = logging.getLogger(__name__)


class ThemeType(Enum):
    """Available theme types"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


class ColorRole(Enum):
    """Color roles in the theme system"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ACCENT = "accent"
    BACKGROUND = "background"
    SURFACE = "surface"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    INFO = "info"
    TEXT_PRIMARY = "text_primary"
    TEXT_SECONDARY = "text_secondary"
    TEXT_DISABLED = "text_disabled"
    BORDER = "border"
    SHADOW = "shadow"
    HIGHLIGHT = "highlight"


@dataclass
class ColorPalette:
    """Color palette for theme"""
    primary: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0)
    secondary: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    accent: Tuple[float, float, float, float] = (1.0, 0.4, 0.0, 1.0)
    background: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
    surface: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    error: Tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0)
    warning: Tuple[float, float, float, float] = (1.0, 0.7, 0.0, 1.0)
    success: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0)
    info: Tuple[float, float, float, float] = (0.2, 0.7, 1.0, 1.0)
    text_primary: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    text_secondary: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    text_disabled: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)
    border: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    shadow: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.3)
    highlight: Tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.3)
    
    def get_color(self, role: ColorRole) -> Tuple[float, float, float, float]:
        """Get color by role"""
        role_map = {
            ColorRole.PRIMARY: self.primary,
            ColorRole.SECONDARY: self.secondary,
            ColorRole.ACCENT: self.accent,
            ColorRole.BACKGROUND: self.background,
            ColorRole.SURFACE: self.surface,
            ColorRole.ERROR: self.error,
            ColorRole.WARNING: self.warning,
            ColorRole.SUCCESS: self.success,
            ColorRole.INFO: self.info,
            ColorRole.TEXT_PRIMARY: self.text_primary,
            ColorRole.TEXT_SECONDARY: self.text_secondary,
            ColorRole.TEXT_DISABLED: self.text_disabled,
            ColorRole.BORDER: self.border,
            ColorRole.SHADOW: self.shadow,
            ColorRole.HIGHLIGHT: self.highlight
        }
        return role_map.get(role, self.primary)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert palette to dictionary"""
        return {
            'primary': list(self.primary),
            'secondary': list(self.secondary),
            'accent': list(self.accent),
            'background': list(self.background),
            'surface': list(self.surface),
            'error': list(self.error),
            'warning': list(self.warning),
            'success': list(self.success),
            'info': list(self.info),
            'text_primary': list(self.text_primary),
            'text_secondary': list(self.text_secondary),
            'text_disabled': list(self.text_disabled),
            'border': list(self.border),
            'shadow': list(self.shadow),
            'highlight': list(self.highlight)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> 'ColorPalette':
        """Create palette from dictionary"""
        return cls(
            primary=tuple(data.get('primary', [0.2, 0.6, 1.0, 1.0])),
            secondary=tuple(data.get('secondary', [0.6, 0.6, 0.6, 1.0])),
            accent=tuple(data.get('accent', [1.0, 0.4, 0.0, 1.0])),
            background=tuple(data.get('background', [0.1, 0.1, 0.1, 1.0])),
            surface=tuple(data.get('surface', [0.2, 0.2, 0.2, 1.0])),
            error=tuple(data.get('error', [0.9, 0.2, 0.2, 1.0])),
            warning=tuple(data.get('warning', [1.0, 0.7, 0.0, 1.0])),
            success=tuple(data.get('success', [0.2, 0.8, 0.2, 1.0])),
            info=tuple(data.get('info', [0.2, 0.7, 1.0, 1.0])),
            text_primary=tuple(data.get('text_primary', [1.0, 1.0, 1.0, 1.0])),
            text_secondary=tuple(data.get('text_secondary', [0.7, 0.7, 0.7, 1.0])),
            text_disabled=tuple(data.get('text_disabled', [0.4, 0.4, 0.4, 1.0])),
            border=tuple(data.get('border', [0.3, 0.3, 0.3, 1.0])),
            shadow=tuple(data.get('shadow', [0.0, 0.0, 0.0, 0.3])),
            highlight=tuple(data.get('highlight', [0.3, 0.6, 1.0, 0.3]))
        )


@dataclass
class ThemeConfig:
    """Configuration for theme system"""
    auto_switch_enabled: bool = True
    transition_duration: float = 0.3
    save_user_preference: bool = True
    config_file_path: str = "config/theme_config.json"
    enable_animations: bool = True
    accessibility_mode: bool = False
    high_contrast_ratio: float = 4.5  # WCAG AA compliance
    
    def __post_init__(self):
        """Validate configuration"""
        if self.transition_duration < 0.1 or self.transition_duration > 2.0:
            raise ValueError("Transition duration out of range")
        if self.high_contrast_ratio < 3.0:
            raise ValueError("Contrast ratio too low for accessibility")


class Theme:
    """Base theme class"""
    
    def __init__(self, name: str, theme_type: ThemeType, palette: ColorPalette, 
                 font_sizes: Optional[Dict[str, float]] = None):
        self.name = name
        self.theme_type = theme_type
        self.palette = palette
        self.font_sizes = font_sizes or self._default_font_sizes()
        self.metadata = {}
        
        # Spacing and sizing
        self.spacing = {
            'xs': dp(4),
            'sm': dp(8),
            'md': dp(16),
            'lg': dp(24),
            'xl': dp(32)
        }
        
        self.border_radius = {
            'none': 0,
            'sm': dp(4),
            'md': dp(8),
            'lg': dp(16),
            'full': dp(9999)
        }
        
        self.shadows = {
            'none': (0, 0, 0, 0),
            'sm': (0, dp(1), dp(3), 0.12),
            'md': (0, dp(4), dp(6), 0.15),
            'lg': (0, dp(10), dp(15), 0.20),
            'xl': (0, dp(25), dp(50), 0.25)
        }
    
    def _default_font_sizes(self) -> Dict[str, float]:
        """Default font sizes for theme"""
        return {
            'caption': dp(10),
            'small': dp(12),
            'body': dp(14),
            'subtitle': dp(16),
            'title': dp(18),
            'heading': dp(24),
            'display': dp(32)
        }
    
    def get_color(self, role: ColorRole) -> Tuple[float, float, float, float]:
        """Get color by role"""
        return self.palette.get_color(role)
    
    def get_font_size(self, size_name: str) -> float:
        """Get font size by name"""
        return self.font_sizes.get(size_name, self.font_sizes['body'])
    
    def get_spacing(self, size_name: str) -> float:
        """Get spacing by name"""
        return self.spacing.get(size_name, self.spacing['md'])
    
    def get_border_radius(self, size_name: str) -> float:
        """Get border radius by name"""
        return self.border_radius.get(size_name, self.border_radius['md'])
    
    def get_shadow(self, size_name: str) -> Tuple[float, float, float, float]:
        """Get shadow by name"""
        return self.shadows.get(size_name, self.shadows['none'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert theme to dictionary"""
        return {
            'name': self.name,
            'theme_type': self.theme_type.value,
            'palette': self.palette.to_dict(),
            'font_sizes': self.font_sizes,
            'spacing': self.spacing,
            'border_radius': self.border_radius,
            'shadows': self.shadows,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        """Create theme from dictionary"""
        palette = ColorPalette.from_dict(data.get('palette', {}))
        theme_type = ThemeType(data.get('theme_type', 'dark'))
        
        theme = cls(
            name=data.get('name', 'Unnamed'),
            theme_type=theme_type,
            palette=palette,
            font_sizes=data.get('font_sizes')
        )
        
        theme.spacing = data.get('spacing', theme.spacing)
        theme.border_radius = data.get('border_radius', theme.border_radius)
        theme.shadows = data.get('shadows', theme.shadows)
        theme.metadata = data.get('metadata', {})
        
        return theme


class DarkTheme(Theme):
    """Dark theme optimized for low-light use"""
    
    def __init__(self):
        palette = ColorPalette(
            primary=(0.2, 0.6, 1.0, 1.0),
            secondary=(0.6, 0.6, 0.6, 1.0),
            accent=(0.0, 0.8, 0.4, 1.0),
            background=(0.08, 0.08, 0.08, 1.0),
            surface=(0.15, 0.15, 0.15, 1.0),
            error=(0.9, 0.3, 0.3, 1.0),
            warning=(1.0, 0.7, 0.0, 1.0),
            success=(0.2, 0.8, 0.3, 1.0),
            info=(0.3, 0.7, 1.0, 1.0),
            text_primary=(0.95, 0.95, 0.95, 1.0),
            text_secondary=(0.7, 0.7, 0.7, 1.0),
            text_disabled=(0.4, 0.4, 0.4, 1.0),
            border=(0.25, 0.25, 0.25, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.5),
            highlight=(0.3, 0.6, 1.0, 0.2)
        )
        
        super().__init__("Dark", ThemeType.DARK, palette)
        self.metadata = {
            'description': 'Dark theme optimized for low-light environments',
            'suitable_for': ['night_use', 'energy_saving', 'eye_strain_reduction']
        }


class LightTheme(Theme):
    """Light theme optimized for bright environments"""
    
    def __init__(self):
        palette = ColorPalette(
            primary=(0.1, 0.4, 0.8, 1.0),
            secondary=(0.4, 0.4, 0.4, 1.0),
            accent=(0.8, 0.3, 0.0, 1.0),
            background=(0.98, 0.98, 0.98, 1.0),
            surface=(1.0, 1.0, 1.0, 1.0),
            error=(0.8, 0.1, 0.1, 1.0),
            warning=(0.9, 0.6, 0.0, 1.0),
            success=(0.1, 0.6, 0.1, 1.0),
            info=(0.1, 0.5, 0.8, 1.0),
            text_primary=(0.1, 0.1, 0.1, 1.0),
            text_secondary=(0.4, 0.4, 0.4, 1.0),
            text_disabled=(0.6, 0.6, 0.6, 1.0),
            border=(0.8, 0.8, 0.8, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.15),
            highlight=(0.1, 0.4, 0.8, 0.1)
        )
        
        super().__init__("Light", ThemeType.LIGHT, palette)
        self.metadata = {
            'description': 'Light theme optimized for bright environments',
            'suitable_for': ['daylight_use', 'outdoor_use', 'high_ambient_light']
        }


class HighContrastTheme(Theme):
    """High contrast theme for accessibility"""
    
    def __init__(self):
        palette = ColorPalette(
            primary=(0.0, 0.6, 1.0, 1.0),
            secondary=(0.8, 0.8, 0.8, 1.0),
            accent=(1.0, 1.0, 0.0, 1.0),
            background=(0.0, 0.0, 0.0, 1.0),
            surface=(0.1, 0.1, 0.1, 1.0),
            error=(1.0, 0.0, 0.0, 1.0),
            warning=(1.0, 1.0, 0.0, 1.0),
            success=(0.0, 1.0, 0.0, 1.0),
            info=(0.0, 1.0, 1.0, 1.0),
            text_primary=(1.0, 1.0, 1.0, 1.0),
            text_secondary=(0.9, 0.9, 0.9, 1.0),
            text_disabled=(0.6, 0.6, 0.6, 1.0),
            border=(1.0, 1.0, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.8),
            highlight=(1.0, 1.0, 0.0, 0.3)
        )
        
        super().__init__("High Contrast", ThemeType.HIGH_CONTRAST, palette)
        self.metadata = {
            'description': 'High contrast theme for accessibility',
            'suitable_for': ['vision_impaired', 'accessibility', 'high_contrast_needed'],
            'contrast_ratio': 7.0  # WCAG AAA compliance
        }


class ThemeManager(EventDispatcher):
    """Manages theme switching and persistence"""
    
    def __init__(self, config: Optional[ThemeConfig] = None):
        super().__init__()
        self.config = config or ThemeConfig()
        
        # Available themes
        self.themes: Dict[str, Theme] = {}
        self.current_theme: Optional[Theme] = None
        
        # Theme switching
        self.theme_change_callbacks: List[Callable[[Theme], None]] = []
        self.auto_theme_enabled = self.config.auto_switch_enabled
        
        # Register default themes
        self._register_default_themes()
        
        # Load user preferences
        self._load_user_preferences()
        
        logger.info("ThemeManager initialized")
    
    def _register_default_themes(self):
        """Register built-in themes"""
        self.register_theme(DarkTheme())
        self.register_theme(LightTheme())
        self.register_theme(HighContrastTheme())
        
        # Set default theme
        if not self.current_theme:
            self.set_theme("Dark")
    
    def register_theme(self, theme: Theme):
        """Register a theme"""
        self.themes[theme.name] = theme
        logger.info(f"Registered theme: {theme.name}")
    
    def set_theme(self, theme_name: str) -> bool:
        """Set active theme by name"""
        if theme_name not in self.themes:
            logger.error(f"Theme '{theme_name}' not found")
            return False
        
        old_theme = self.current_theme
        new_theme = self.themes[theme_name]
        
        self.current_theme = new_theme
        
        # Execute callbacks
        for callback in self.theme_change_callbacks:
            try:
                callback(new_theme)
            except Exception as e:
                logger.error(f"Theme change callback error: {e}")
        
        # Save preference
        if self.config.save_user_preference:
            self._save_user_preference(theme_name)
        
        # Dispatch event
        self.dispatch('on_theme_changed', old_theme, new_theme)
        
        logger.info(f"Theme changed to: {theme_name}")
        return True
    
    def get_current_theme(self) -> Optional[Theme]:
        """Get current active theme"""
        return self.current_theme
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self.themes.keys())
    
    def add_theme_change_callback(self, callback: Callable[[Theme], None]):
        """Add callback for theme changes"""
        self.theme_change_callbacks.append(callback)
    
    def remove_theme_change_callback(self, callback: Callable[[Theme], None]):
        """Remove theme change callback"""
        if callback in self.theme_change_callbacks:
            self.theme_change_callbacks.remove(callback)
    
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        if not self.current_theme:
            return
        
        if self.current_theme.theme_type == ThemeType.DARK:
            self.set_theme("Light")
        else:
            self.set_theme("Dark")
    
    def enable_auto_theme(self, enabled: bool = True):
        """Enable or disable automatic theme switching"""
        self.auto_theme_enabled = enabled
        self.config.auto_switch_enabled = enabled
        
        if enabled:
            # Could implement time-based or sensor-based switching here
            pass
    
    def create_custom_theme(self, name: str, base_theme: str, 
                          color_overrides: Dict[str, Tuple[float, float, float, float]]) -> bool:
        """Create custom theme based on existing theme"""
        if base_theme not in self.themes:
            logger.error(f"Base theme '{base_theme}' not found")
            return False
        
        base = self.themes[base_theme]
        custom_palette = ColorPalette(**base.palette.__dict__)
        
        # Apply color overrides
        for color_name, color_value in color_overrides.items():
            if hasattr(custom_palette, color_name):
                setattr(custom_palette, color_name, color_value)
        
        custom_theme = Theme(name, ThemeType.CUSTOM, custom_palette, base.font_sizes)
        self.register_theme(custom_theme)
        
        logger.info(f"Created custom theme: {name}")
        return True
    
    def export_theme(self, theme_name: str, file_path: str) -> bool:
        """Export theme to file"""
        if theme_name not in self.themes:
            return False
        
        try:
            theme_data = self.themes[theme_name].to_dict()
            with open(file_path, 'w') as f:
                json.dump(theme_data, f, indent=2)
            
            logger.info(f"Theme exported to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Theme export failed: {e}")
            return False
    
    def import_theme(self, file_path: str) -> bool:
        """Import theme from file"""
        try:
            with open(file_path, 'r') as f:
                theme_data = json.load(f)
            
            theme = Theme.from_dict(theme_data)
            self.register_theme(theme)
            
            logger.info(f"Theme imported from: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Theme import failed: {e}")
            return False
    
    def _save_user_preference(self, theme_name: str):
        """Save user theme preference"""
        try:
            os.makedirs(os.path.dirname(self.config.config_file_path), exist_ok=True)
            
            config_data = {
                'current_theme': theme_name,
                'auto_theme_enabled': self.auto_theme_enabled,
                'last_updated': time.time()
            }
            
            with open(self.config.config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save theme preference: {e}")
    
    def _load_user_preferences(self):
        """Load user theme preferences"""
        try:
            if os.path.exists(self.config.config_file_path):
                with open(self.config.config_file_path, 'r') as f:
                    config_data = json.load(f)
                
                preferred_theme = config_data.get('current_theme', 'Dark')
                self.auto_theme_enabled = config_data.get('auto_theme_enabled', True)
                
                if preferred_theme in self.themes:
                    self.set_theme(preferred_theme)
                    
        except Exception as e:
            logger.error(f"Failed to load theme preferences: {e}")
    
    def get_theme_stats(self) -> Dict[str, Any]:
        """Get theme manager statistics"""
        return {
            'current_theme': self.current_theme.name if self.current_theme else None,
            'available_themes': len(self.themes),
            'auto_theme_enabled': self.auto_theme_enabled,
            'callbacks_registered': len(self.theme_change_callbacks)
        }
    
    # Event definitions
    __events__ = ('on_theme_changed',)
    
    def on_theme_changed(self, old_theme, new_theme):
        """Called when theme changes"""
        pass


def create_theme_manager(config: Optional[ThemeConfig] = None) -> ThemeManager:
    """Factory function to create theme manager"""
    return ThemeManager(config or ThemeConfig())


def create_dark_theme() -> DarkTheme:
    """Factory function to create dark theme"""
    return DarkTheme()


def create_light_theme() -> LightTheme:
    """Factory function to create light theme"""
    return LightTheme()


def create_high_contrast_theme() -> HighContrastTheme:
    """Factory function to create high contrast theme"""
    return HighContrastTheme()


if __name__ == '__main__':
    # Test theme system
    theme_manager = create_theme_manager()
    
    print("Theme Manager Test")
    print(f"Available themes: {theme_manager.get_available_themes()}")
    print(f"Current theme: {theme_manager.get_current_theme().name}")
    
    # Test theme switching
    theme_manager.set_theme("Light")
    print(f"Switched to: {theme_manager.get_current_theme().name}")
    
    # Test color retrieval
    current_theme = theme_manager.get_current_theme()
    primary_color = current_theme.get_color(ColorRole.PRIMARY)
    print(f"Primary color: {primary_color}")
    
    print("Theme system test completed")