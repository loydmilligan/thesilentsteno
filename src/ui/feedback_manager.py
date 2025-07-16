"""
Visual and Haptic Feedback Management

This module manages user feedback including visual effects, audio cues,
and haptic responses for touch interactions in The Silent Steno.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from enum import Enum
import threading
import time
import logging
import queue
import os

try:
    from kivy.event import EventDispatcher
    from kivy.clock import Clock
    from kivy.animation import Animation
    from kivy.graphics import Color, Rectangle, Line, Ellipse
    from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Scale, Translate
    from kivy.metrics import dp
    from kivy.core.audio import SoundLoader
    from kivy.uix.widget import Widget
except ImportError:
    raise ImportError("Kivy not available. Install with: pip install kivy")

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    VISUAL = "visual"
    AUDIO = "audio"
    HAPTIC = "haptic"
    COMBINED = "combined"


class FeedbackEvent(Enum):
    """Feedback events"""
    TOUCH_DOWN = "touch_down"
    TOUCH_UP = "touch_up"
    BUTTON_PRESS = "button_press"
    NAVIGATION = "navigation"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    SCROLL = "scroll"
    SELECTION = "selection"


class VisualEffectType(Enum):
    """Types of visual effects"""
    RIPPLE = "ripple"
    HIGHLIGHT = "highlight"
    SHAKE = "shake"
    PULSE = "pulse"
    GLOW = "glow"
    FADE = "fade"
    SCALE = "scale"
    FLASH = "flash"


@dataclass
class FeedbackConfig:
    """Configuration for feedback system"""
    enable_visual: bool = True
    enable_audio: bool = True
    enable_haptic: bool = True
    visual_intensity: float = 1.0  # 0.0 to 1.0
    audio_volume: float = 0.5     # 0.0 to 1.0
    haptic_intensity: float = 0.8  # 0.0 to 1.0
    animation_duration: float = 0.3
    ripple_duration: float = 0.4
    pulse_duration: float = 0.6
    max_concurrent_effects: int = 5
    sound_cache_size: int = 20
    accessibility_mode: bool = False
    reduced_motion: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        self.visual_intensity = max(0.0, min(1.0, self.visual_intensity))
        self.audio_volume = max(0.0, min(1.0, self.audio_volume))
        self.haptic_intensity = max(0.0, min(1.0, self.haptic_intensity))


@dataclass
class VisualEffect:
    """Definition of a visual effect"""
    effect_type: VisualEffectType
    widget: Widget
    position: Optional[Tuple[float, float]] = None
    size: Optional[Tuple[float, float]] = None
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.3)
    duration: float = 0.3
    start_time: float = field(default_factory=time.time)
    animation: Optional[Animation] = None
    graphics_instructions: List[Any] = field(default_factory=list)
    
    def is_active(self) -> bool:
        """Check if effect is still active"""
        return time.time() - self.start_time < self.duration


@dataclass
class AudioCue:
    """Definition of an audio cue"""
    sound_file: str
    volume: float = 1.0
    loop: bool = False
    start_time: float = field(default_factory=time.time)
    sound_object: Optional[Any] = None


@dataclass 
class HapticFeedback:
    """Definition of haptic feedback"""
    pattern: str  # 'short', 'medium', 'long', 'double', 'triple'
    intensity: float = 1.0
    duration: float = 0.1
    start_time: float = field(default_factory=time.time)


class VisualFeedback:
    """Manages visual feedback effects"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.active_effects: List[VisualEffect] = []
        self.effect_cleanup_interval = 0.1
        
        # Start cleanup scheduler
        Clock.schedule_interval(self._cleanup_effects, self.effect_cleanup_interval)
        
        logger.debug("VisualFeedback initialized")
    
    def create_ripple(self, widget: Widget, touch_pos: Tuple[float, float], 
                     color: Optional[Tuple[float, float, float, float]] = None) -> VisualEffect:
        """Create ripple effect at touch position"""
        if not self.config.enable_visual or self.config.reduced_motion:
            return None
        
        color = color or (1.0, 1.0, 1.0, 0.3 * self.config.visual_intensity)
        
        effect = VisualEffect(
            effect_type=VisualEffectType.RIPPLE,
            widget=widget,
            position=touch_pos,
            color=color,
            duration=self.config.ripple_duration
        )
        
        self._start_ripple_animation(effect)
        self.active_effects.append(effect)
        
        return effect
    
    def create_highlight(self, widget: Widget, 
                        color: Optional[Tuple[float, float, float, float]] = None) -> VisualEffect:
        """Create highlight effect on widget"""
        if not self.config.enable_visual:
            return None
        
        color = color or (0.3, 0.6, 1.0, 0.2 * self.config.visual_intensity)
        
        effect = VisualEffect(
            effect_type=VisualEffectType.HIGHLIGHT,
            widget=widget,
            color=color,
            duration=self.config.animation_duration
        )
        
        self._start_highlight_animation(effect)
        self.active_effects.append(effect)
        
        return effect
    
    def create_pulse(self, widget: Widget, 
                    color: Optional[Tuple[float, float, float, float]] = None) -> VisualEffect:
        """Create pulse effect on widget"""
        if not self.config.enable_visual or self.config.reduced_motion:
            return None
        
        color = color or (1.0, 1.0, 1.0, 0.4 * self.config.visual_intensity)
        
        effect = VisualEffect(
            effect_type=VisualEffectType.PULSE,
            widget=widget,
            color=color,
            duration=self.config.pulse_duration
        )
        
        self._start_pulse_animation(effect)
        self.active_effects.append(effect)
        
        return effect
    
    def create_shake(self, widget: Widget) -> VisualEffect:
        """Create shake effect on widget"""
        if not self.config.enable_visual or self.config.reduced_motion:
            return None
        
        effect = VisualEffect(
            effect_type=VisualEffectType.SHAKE,
            widget=widget,
            duration=0.5
        )
        
        self._start_shake_animation(effect)
        self.active_effects.append(effect)
        
        return effect
    
    def create_flash(self, widget: Widget,
                    color: Optional[Tuple[float, float, float, float]] = None) -> VisualEffect:
        """Create flash effect on widget"""
        if not self.config.enable_visual:
            return None
        
        color = color or (1.0, 1.0, 1.0, 0.8 * self.config.visual_intensity)
        
        effect = VisualEffect(
            effect_type=VisualEffectType.FLASH,
            widget=widget,
            color=color,
            duration=0.2
        )
        
        self._start_flash_animation(effect)
        self.active_effects.append(effect)
        
        return effect
    
    def _start_ripple_animation(self, effect: VisualEffect):
        """Start ripple animation"""
        widget = effect.widget
        pos = effect.position
        
        with widget.canvas.after:
            Color(*effect.color)
            ripple = Ellipse(pos=pos, size=(0, 0))
            effect.graphics_instructions.append(ripple)
        
        # Animate ripple expansion
        target_size = max(widget.width, widget.height) * 2
        target_pos = (pos[0] - target_size/2, pos[1] - target_size/2)
        
        anim = Animation(
            size=(target_size, target_size),
            pos=target_pos,
            duration=effect.duration,
            t='out_quad'
        )
        
        effect.animation = anim
        anim.start(ripple)
    
    def _start_highlight_animation(self, effect: VisualEffect):
        """Start highlight animation"""
        widget = effect.widget
        
        with widget.canvas.after:
            Color(*effect.color)
            highlight = Rectangle(pos=widget.pos, size=widget.size)
            effect.graphics_instructions.append(highlight)
        
        # Animate highlight fade
        anim = Animation(
            rgba=(effect.color[0], effect.color[1], effect.color[2], 0),
            duration=effect.duration,
            t='out_quad'
        )
        
        effect.animation = anim
        anim.start(widget.canvas.after.children[-2])  # Color instruction
    
    def _start_pulse_animation(self, effect: VisualEffect):
        """Start pulse animation"""
        widget = effect.widget
        
        with widget.canvas.after:
            Color(*effect.color)
            pulse = Rectangle(pos=widget.pos, size=widget.size)
            effect.graphics_instructions.append(pulse)
        
        # Create pulsing animation
        anim = (Animation(rgba=(effect.color[0], effect.color[1], effect.color[2], 0), 
                         duration=effect.duration/2, t='in_out_quad') + 
                Animation(rgba=effect.color, duration=effect.duration/2, t='in_out_quad'))
        
        effect.animation = anim
        anim.start(widget.canvas.after.children[-2])  # Color instruction
    
    def _start_shake_animation(self, effect: VisualEffect):
        """Start shake animation"""
        widget = effect.widget
        original_x = widget.x
        
        # Create shake sequence
        shake_distance = dp(5)
        shake_duration = 0.1
        
        anim = (Animation(x=original_x + shake_distance, duration=shake_duration) +
                Animation(x=original_x - shake_distance, duration=shake_duration) +
                Animation(x=original_x + shake_distance, duration=shake_duration) +
                Animation(x=original_x - shake_distance, duration=shake_duration) +
                Animation(x=original_x, duration=shake_duration))
        
        effect.animation = anim
        anim.start(widget)
    
    def _start_flash_animation(self, effect: VisualEffect):
        """Start flash animation"""
        widget = effect.widget
        
        with widget.canvas.after:
            Color(*effect.color)
            flash = Rectangle(pos=widget.pos, size=widget.size)
            effect.graphics_instructions.append(flash)
        
        # Animate flash fade
        anim = Animation(
            rgba=(effect.color[0], effect.color[1], effect.color[2], 0),
            duration=effect.duration,
            t='out_expo'
        )
        
        effect.animation = anim
        anim.start(widget.canvas.after.children[-2])  # Color instruction
    
    def _cleanup_effects(self, dt):
        """Clean up finished effects"""
        current_time = time.time()
        
        # Remove finished effects
        finished_effects = [e for e in self.active_effects if not e.is_active()]
        
        for effect in finished_effects:
            self._remove_effect(effect)
        
        # Limit concurrent effects
        if len(self.active_effects) > self.config.max_concurrent_effects:
            # Remove oldest effects
            excess_count = len(self.active_effects) - self.config.max_concurrent_effects
            for effect in self.active_effects[:excess_count]:
                self._remove_effect(effect)
    
    def _remove_effect(self, effect: VisualEffect):
        """Remove effect and clean up graphics"""
        if effect in self.active_effects:
            self.active_effects.remove(effect)
        
        # Stop animation
        if effect.animation:
            effect.animation.stop_all(effect.widget)
        
        # Remove graphics instructions
        for instruction in effect.graphics_instructions:
            if instruction.parent:
                instruction.parent.remove(instruction)
    
    def clear_all_effects(self):
        """Clear all active effects"""
        for effect in self.active_effects[:]:
            self._remove_effect(effect)


class AudioFeedback:
    """Manages audio feedback"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.sound_cache: Dict[str, Any] = {}
        self.active_sounds: List[AudioCue] = []
        
        # Default sound mappings
        self.sound_mappings = {
            FeedbackEvent.BUTTON_PRESS: "sounds/click.wav",
            FeedbackEvent.NAVIGATION: "sounds/swoosh.wav",
            FeedbackEvent.ERROR: "sounds/error.wav",
            FeedbackEvent.SUCCESS: "sounds/success.wav",
            FeedbackEvent.WARNING: "sounds/warning.wav",
            FeedbackEvent.LONG_PRESS: "sounds/long_press.wav"
        }
        
        # Preload common sounds
        self._preload_sounds()
        
        logger.debug("AudioFeedback initialized")
    
    def _preload_sounds(self):
        """Preload commonly used sounds"""
        for event, sound_file in self.sound_mappings.items():
            self._load_sound(sound_file)
    
    def _load_sound(self, sound_file: str):
        """Load sound file into cache"""
        if sound_file in self.sound_cache:
            return self.sound_cache[sound_file]
        
        try:
            if os.path.exists(sound_file):
                sound = SoundLoader.load(sound_file)
                if sound:
                    self.sound_cache[sound_file] = sound
                    return sound
            else:
                logger.warning(f"Sound file not found: {sound_file}")
        except Exception as e:
            logger.error(f"Failed to load sound {sound_file}: {e}")
        
        return None
    
    def play_sound(self, event: FeedbackEvent, volume: Optional[float] = None):
        """Play sound for feedback event"""
        if not self.config.enable_audio:
            return
        
        sound_file = self.sound_mappings.get(event)
        if not sound_file:
            return
        
        sound = self._load_sound(sound_file)
        if not sound:
            return
        
        # Set volume
        final_volume = (volume or 1.0) * self.config.audio_volume
        sound.volume = final_volume
        
        # Play sound
        sound.play()
        
        # Track active sound
        cue = AudioCue(
            sound_file=sound_file,
            volume=final_volume,
            sound_object=sound
        )
        self.active_sounds.append(cue)
        
        # Schedule cleanup
        Clock.schedule_once(lambda dt: self._cleanup_sound(cue), sound.length)
    
    def play_custom_sound(self, sound_file: str, volume: Optional[float] = None, loop: bool = False):
        """Play custom sound file"""
        if not self.config.enable_audio:
            return
        
        sound = self._load_sound(sound_file)
        if not sound:
            return
        
        final_volume = (volume or 1.0) * self.config.audio_volume
        sound.volume = final_volume
        sound.loop = loop
        
        sound.play()
        
        cue = AudioCue(
            sound_file=sound_file,
            volume=final_volume,
            loop=loop,
            sound_object=sound
        )
        self.active_sounds.append(cue)
        
        if not loop:
            Clock.schedule_once(lambda dt: self._cleanup_sound(cue), sound.length)
    
    def _cleanup_sound(self, cue: AudioCue):
        """Clean up finished sound"""
        if cue in self.active_sounds:
            self.active_sounds.remove(cue)
    
    def stop_all_sounds(self):
        """Stop all playing sounds"""
        for cue in self.active_sounds[:]:
            if cue.sound_object:
                cue.sound_object.stop()
            self._cleanup_sound(cue)
    
    def set_sound_mapping(self, event: FeedbackEvent, sound_file: str):
        """Set custom sound for event"""
        self.sound_mappings[event] = sound_file


class HapticFeedback:
    """Manages haptic feedback (vibration)"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.vibration_patterns = {
            'short': [100],
            'medium': [200],
            'long': [500],
            'double': [100, 50, 100],
            'triple': [100, 50, 100, 50, 100],
            'error': [200, 100, 200],
            'success': [50, 50, 50]
        }
        
        # Try to import platform-specific vibration
        self.vibrator = None
        self._initialize_vibration()
        
        logger.debug("HapticFeedback initialized")
    
    def _initialize_vibration(self):
        """Initialize platform-specific vibration"""
        try:
            # Android vibration
            from jnius import autoclass, PythonJavaClass
            Context = autoclass('android.content.Context')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.vibrator = PythonActivity.mActivity.getSystemService(Context.VIBRATOR_SERVICE)
        except:
            # iOS or other platforms could be added here
            logger.debug("Platform vibration not available")
    
    def vibrate(self, pattern: str = 'short', intensity: Optional[float] = None):
        """Trigger haptic feedback"""
        if not self.config.enable_haptic:
            return
        
        if pattern not in self.vibration_patterns:
            pattern = 'short'
        
        final_intensity = (intensity or 1.0) * self.config.haptic_intensity
        durations = self.vibration_patterns[pattern]
        
        # Scale durations by intensity
        scaled_durations = [int(d * final_intensity) for d in durations]
        
        self._execute_vibration(scaled_durations)
    
    def _execute_vibration(self, durations: List[int]):
        """Execute vibration pattern"""
        if self.vibrator:
            try:
                # Android vibration
                if len(durations) == 1:
                    self.vibrator.vibrate(durations[0])
                else:
                    # Pattern vibration: [delay, vibrate, delay, vibrate, ...]
                    pattern = []
                    for i, duration in enumerate(durations):
                        if i > 0:
                            pattern.append(50)  # Small delay between vibrations
                        pattern.append(duration)
                    
                    from jnius import autoclass
                    self.vibrator.vibrate(pattern, -1)  # -1 means don't repeat
            except Exception as e:
                logger.error(f"Vibration failed: {e}")
        else:
            # Fallback: log vibration for debugging
            logger.debug(f"Haptic feedback: {durations}ms")


class FeedbackManager(EventDispatcher):
    """Main feedback manager coordinating all feedback types"""
    
    def __init__(self, config: Optional[FeedbackConfig] = None):
        super().__init__()
        self.config = config or FeedbackConfig()
        
        # Initialize feedback components
        self.visual = VisualFeedback(self.config)
        self.audio = AudioFeedback(self.config)
        self.haptic = HapticFeedback(self.config)
        
        # Event mappings
        self.feedback_mappings: Dict[FeedbackEvent, Dict[FeedbackType, Any]] = {
            FeedbackEvent.BUTTON_PRESS: {
                FeedbackType.VISUAL: ('ripple', {}),
                FeedbackType.AUDIO: ('button_press', {}),
                FeedbackType.HAPTIC: ('short', {})
            },
            FeedbackEvent.LONG_PRESS: {
                FeedbackType.VISUAL: ('pulse', {}),
                FeedbackType.AUDIO: ('long_press', {}),
                FeedbackType.HAPTIC: ('medium', {})
            },
            FeedbackEvent.ERROR: {
                FeedbackType.VISUAL: ('shake', {}),
                FeedbackType.AUDIO: ('error', {}),
                FeedbackType.HAPTIC: ('error', {})
            },
            FeedbackEvent.SUCCESS: {
                FeedbackType.VISUAL: ('highlight', {'color': (0.2, 0.8, 0.2, 0.3)}),
                FeedbackType.AUDIO: ('success', {}),
                FeedbackType.HAPTIC: ('success', {})
            }
        }
        
        logger.info("FeedbackManager initialized")
    
    def provide_feedback(self, event: FeedbackEvent, widget: Optional[Widget] = None,
                        feedback_types: Optional[List[FeedbackType]] = None,
                        **kwargs):
        """Provide feedback for an event"""
        
        # Default to combined feedback
        if feedback_types is None:
            feedback_types = [FeedbackType.VISUAL, FeedbackType.AUDIO, FeedbackType.HAPTIC]
        
        # Get feedback configuration for event
        event_config = self.feedback_mappings.get(event, {})
        
        for feedback_type in feedback_types:
            if feedback_type in event_config:
                self._execute_feedback(feedback_type, event_config[feedback_type], widget, **kwargs)
    
    def _execute_feedback(self, feedback_type: FeedbackType, config: tuple, 
                         widget: Optional[Widget], **kwargs):
        """Execute specific type of feedback"""
        method_name, default_args = config
        
        # Merge default args with provided kwargs
        args = {**default_args, **kwargs}
        
        try:
            if feedback_type == FeedbackType.VISUAL and widget:
                if method_name == 'ripple':
                    touch_pos = args.get('touch_pos', widget.center)
                    self.visual.create_ripple(widget, touch_pos, args.get('color'))
                elif method_name == 'highlight':
                    self.visual.create_highlight(widget, args.get('color'))
                elif method_name == 'pulse':
                    self.visual.create_pulse(widget, args.get('color'))
                elif method_name == 'shake':
                    self.visual.create_shake(widget)
                elif method_name == 'flash':
                    self.visual.create_flash(widget, args.get('color'))
            
            elif feedback_type == FeedbackType.AUDIO:
                if hasattr(FeedbackEvent, method_name.upper()):
                    event_enum = FeedbackEvent(method_name)
                    self.audio.play_sound(event_enum, args.get('volume'))
                else:
                    # Custom sound file
                    self.audio.play_custom_sound(method_name, args.get('volume'))
            
            elif feedback_type == FeedbackType.HAPTIC:
                self.haptic.vibrate(method_name, args.get('intensity'))
                
        except Exception as e:
            logger.error(f"Feedback execution failed: {e}")
    
    def create_ripple(self, widget: Widget, touch_pos: Tuple[float, float], **kwargs):
        """Convenience method to create ripple effect"""
        return self.visual.create_ripple(widget, touch_pos, kwargs.get('color'))
    
    def play_sound(self, event: FeedbackEvent, **kwargs):
        """Convenience method to play sound"""
        self.audio.play_sound(event, kwargs.get('volume'))
    
    def vibrate(self, pattern: str = 'short', **kwargs):
        """Convenience method to trigger vibration"""
        self.haptic.vibrate(pattern, kwargs.get('intensity'))
    
    def set_config(self, config: FeedbackConfig):
        """Update feedback configuration"""
        self.config = config
        self.visual.config = config
        self.audio.config = config
        self.haptic.config = config
    
    def enable_accessibility_mode(self, enabled: bool = True):
        """Enable accessibility mode with enhanced feedback"""
        self.config.accessibility_mode = enabled
        if enabled:
            # Increase feedback intensity for accessibility
            self.config.visual_intensity = 1.0
            self.config.audio_volume = 0.8
            self.config.haptic_intensity = 1.0
    
    def enable_reduced_motion(self, enabled: bool = True):
        """Enable reduced motion mode"""
        self.config.reduced_motion = enabled
    
    def clear_all_feedback(self):
        """Clear all active feedback"""
        self.visual.clear_all_effects()
        self.audio.stop_all_sounds()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback system statistics"""
        return {
            'active_visual_effects': len(self.visual.active_effects),
            'active_sounds': len(self.audio.active_sounds),
            'sound_cache_size': len(self.audio.sound_cache),
            'config': {
                'visual_enabled': self.config.enable_visual,
                'audio_enabled': self.config.enable_audio,
                'haptic_enabled': self.config.enable_haptic,
                'accessibility_mode': self.config.accessibility_mode,
                'reduced_motion': self.config.reduced_motion
            }
        }


def create_feedback_manager(config: Optional[FeedbackConfig] = None) -> FeedbackManager:
    """Factory function to create feedback manager"""
    return FeedbackManager(config or FeedbackConfig())


def create_accessibility_config() -> FeedbackConfig:
    """Create feedback configuration optimized for accessibility"""
    return FeedbackConfig(
        enable_visual=True,
        enable_audio=True,
        enable_haptic=True,
        visual_intensity=1.0,
        audio_volume=0.8,
        haptic_intensity=1.0,
        accessibility_mode=True,
        animation_duration=0.5,  # Slower animations
        ripple_duration=0.6
    )


def create_minimal_config() -> FeedbackConfig:
    """Create minimal feedback configuration"""
    return FeedbackConfig(
        enable_visual=True,
        enable_audio=False,
        enable_haptic=False,
        visual_intensity=0.5,
        animation_duration=0.2,
        reduced_motion=True
    )


if __name__ == '__main__':
    # Test feedback manager
    config = FeedbackConfig()
    feedback_manager = create_feedback_manager(config)
    
    print("FeedbackManager Test")
    print(f"Stats: {feedback_manager.get_feedback_stats()}")
    
    # Test haptic feedback
    feedback_manager.vibrate('short')
    
    print("Feedback manager test completed")