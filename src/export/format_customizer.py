"""
Format Customization System

User-configurable export templates and format customization for meeting data.
"""

import logging
import json
# import yaml  # Optional dependency
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import re
from datetime import datetime
from string import Template

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Template type enumeration"""
    EMAIL = "email"
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"
    JSON = "json"
    CSV = "csv"

class ContentSection(Enum):
    """Available content sections"""
    HEADER = "header"
    SUMMARY = "summary"
    TRANSCRIPT = "transcript"
    ANALYSIS = "analysis"
    PARTICIPANTS = "participants"
    ACTION_ITEMS = "action_items"
    FOOTER = "footer"
    METADATA = "metadata"

@dataclass
class FormatOptions:
    """Format-specific options"""
    include_timestamps: bool = True
    include_speaker_names: bool = True
    include_confidence_scores: bool = False
    speaker_name_format: str = "{name}:"
    timestamp_format: str = "[{time}]"
    paragraph_spacing: int = 1
    max_line_length: Optional[int] = None
    word_wrap: bool = True
    include_page_numbers: bool = True
    include_table_of_contents: bool = True
    font_family: str = "Arial"
    font_size: int = 12
    date_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = "{:.2f}"

@dataclass
class TemplateConfig:
    """Template configuration"""
    name: str
    description: str
    template_type: TemplateType
    format_options: FormatOptions
    sections: List[ContentSection] = field(default_factory=list)
    custom_fields: Dict[str, str] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class ExportTemplate:
    """Export template with content"""
    config: TemplateConfig
    content_templates: Dict[ContentSection, str] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, str] = field(default_factory=dict)

class FormatCustomizer:
    """Template system for customizing export formats"""
    
    def __init__(self, templates_dir: str = "/tmp/silentSteno_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, ExportTemplate] = {}
        self.default_variables: Dict[str, Any] = {}
        self._load_templates()
        self._initialize_default_templates()
    
    def _load_templates(self):
        """Load templates from disk"""
        try:
            for template_file in self.templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                    
                    template = self._deserialize_template(template_data)
                    self.templates[template.config.name] = template
                    
                except Exception as e:
                    logger.error(f"Failed to load template {template_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.templates)} templates")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {str(e)}")
    
    def _initialize_default_templates(self):
        """Initialize default templates if none exist"""
        if not self.templates:
            # Create default templates
            self._create_default_email_template()
            self._create_default_pdf_template()
            self._create_default_text_template()
            
            logger.info("Created default templates")
    
    def _create_default_email_template(self):
        """Create default email template"""
        config = TemplateConfig(
            name="default_email",
            description="Default email export template",
            template_type=TemplateType.EMAIL,
            format_options=FormatOptions(
                include_timestamps=True,
                include_speaker_names=True,
                timestamp_format="[{time}]",
                speaker_name_format="{name}: "
            ),
            sections=[
                ContentSection.HEADER,
                ContentSection.SUMMARY,
                ContentSection.TRANSCRIPT,
                ContentSection.ACTION_ITEMS,
                ContentSection.FOOTER
            ]
        )
        
        content_templates = {
            ContentSection.HEADER: """
Meeting Recording: ${session_title}
Date: ${date}
Duration: ${duration}
Participants: ${participant_count}

""",
            ContentSection.SUMMARY: """
SUMMARY:
${summary}

""",
            ContentSection.TRANSCRIPT: """
TRANSCRIPT:
${transcript}

""",
            ContentSection.ACTION_ITEMS: """
ACTION ITEMS:
${action_items}

""",
            ContentSection.FOOTER: """
Generated by Silent Steno Device
${generation_time}
"""
        }
        
        template = ExportTemplate(
            config=config,
            content_templates=content_templates
        )
        
        self.templates[config.name] = template
    
    def _create_default_pdf_template(self):
        """Create default PDF template"""
        config = TemplateConfig(
            name="default_pdf",
            description="Default PDF export template",
            template_type=TemplateType.PDF,
            format_options=FormatOptions(
                include_timestamps=True,
                include_speaker_names=True,
                include_page_numbers=True,
                include_table_of_contents=True,
                font_family="Helvetica",
                font_size=12
            ),
            sections=[
                ContentSection.HEADER,
                ContentSection.SUMMARY,
                ContentSection.TRANSCRIPT,
                ContentSection.ANALYSIS,
                ContentSection.PARTICIPANTS,
                ContentSection.METADATA
            ],
            styling={
                "title_color": "#2E4BC6",
                "header_color": "#1E3A8A",
                "text_color": "#000000",
                "background_color": "#FFFFFF"
            }
        )
        
        content_templates = {
            ContentSection.HEADER: """
${session_title}
${date} | ${duration} | ${participant_count} participants
""",
            ContentSection.SUMMARY: """
Executive Summary
${summary}
""",
            ContentSection.TRANSCRIPT: """
Full Transcript
${transcript}
""",
            ContentSection.ANALYSIS: """
Analysis & Insights
Key Topics: ${key_topics}
Sentiment: ${sentiment}
""",
            ContentSection.PARTICIPANTS: """
Participant Statistics
${participant_stats}
""",
            ContentSection.METADATA: """
Document Information
Generated: ${generation_time}
Session ID: ${session_id}
"""
        }
        
        template = ExportTemplate(
            config=config,
            content_templates=content_templates
        )
        
        self.templates[config.name] = template
    
    def _create_default_text_template(self):
        """Create default text template"""
        config = TemplateConfig(
            name="default_text",
            description="Default plain text export template",
            template_type=TemplateType.TEXT,
            format_options=FormatOptions(
                include_timestamps=True,
                include_speaker_names=True,
                max_line_length=80,
                word_wrap=True,
                timestamp_format="[{time}]",
                speaker_name_format="{name}: "
            ),
            sections=[
                ContentSection.HEADER,
                ContentSection.TRANSCRIPT,
                ContentSection.SUMMARY,
                ContentSection.FOOTER
            ]
        )
        
        content_templates = {
            ContentSection.HEADER: """
================================================================================
MEETING RECORDING: ${session_title}
================================================================================
Date: ${date}
Duration: ${duration}
Participants: ${participant_count}

""",
            ContentSection.TRANSCRIPT: """
TRANSCRIPT:
--------------------------------------------------------------------------------
${transcript}

""",
            ContentSection.SUMMARY: """
SUMMARY:
--------------------------------------------------------------------------------
${summary}

""",
            ContentSection.FOOTER: """
================================================================================
Generated by Silent Steno Device on ${generation_time}
Session ID: ${session_id}
================================================================================
"""
        }
        
        template = ExportTemplate(
            config=config,
            content_templates=content_templates
        )
        
        self.templates[config.name] = template
    
    def create_export_template(self, config: TemplateConfig) -> ExportTemplate:
        """Create new export template"""
        template = ExportTemplate(config=config)
        self.templates[config.name] = template
        self.save_template(template)
        
        logger.info(f"Created template: {config.name}")
        return template
    
    def customize_format(self, template_name: str, session_data: Dict[str, Any]) -> str:
        """Customize format using template"""
        if template_name not in self.templates:
            logger.error(f"Template not found: {template_name}")
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        return self._apply_template(template, session_data)
    
    def _apply_template(self, template: ExportTemplate, session_data: Dict[str, Any]) -> str:
        """Apply template to session data"""
        try:
            # Prepare variables
            variables = self._prepare_variables(session_data, template)
            
            # Generate content for each section
            content_parts = []
            
            for section in template.config.sections:
                if section in template.content_templates:
                    section_template = template.content_templates[section]
                    section_content = self._apply_section_template(
                        section_template, variables, template.config.format_options, section
                    )
                    content_parts.append(section_content)
            
            # Combine all sections
            final_content = "".join(content_parts)
            
            # Apply post-processing
            final_content = self._apply_post_processing(final_content, template.config.format_options)
            
            return final_content
            
        except Exception as e:
            logger.error(f"Template application failed: {str(e)}")
            raise
    
    def _prepare_variables(self, session_data: Dict[str, Any], template: ExportTemplate) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        variables = self.default_variables.copy()
        variables.update(template.variables)
        
        # Add session data
        variables.update(session_data)
        
        # Add computed variables
        variables['generation_time'] = datetime.now().strftime(template.config.format_options.date_format)
        
        # Format transcript if present
        if 'transcript' in session_data and isinstance(session_data['transcript'], list):
            formatted_transcript = self._format_transcript(
                session_data['transcript'], 
                template.config.format_options
            )
            variables['transcript'] = formatted_transcript
        
        # Format action items if present
        if 'analysis' in session_data and 'action_items' in session_data['analysis']:
            action_items = session_data['analysis']['action_items']
            if isinstance(action_items, list):
                formatted_actions = '\n'.join([f"• {item}" for item in action_items])
                variables['action_items'] = formatted_actions
        
        # Format participant stats
        if 'participants' in session_data:
            formatted_participants = self._format_participants(
                session_data['participants'],
                template.config.format_options
            )
            variables['participant_stats'] = formatted_participants
        
        # Format key topics
        if 'analysis' in session_data and 'key_topics' in session_data['analysis']:
            key_topics = session_data['analysis']['key_topics']
            if isinstance(key_topics, list):
                variables['key_topics'] = ', '.join(key_topics)
        
        return variables
    
    def _apply_section_template(self, section_template: str, variables: Dict[str, Any], 
                              format_options: FormatOptions, section: ContentSection) -> str:
        """Apply template to a specific section"""
        try:
            # Use string Template for variable substitution
            template = Template(section_template)
            content = template.safe_substitute(variables)
            
            # Apply section-specific formatting
            if section == ContentSection.TRANSCRIPT:
                content = self._format_transcript_section(content, format_options)
            
            return content
            
        except Exception as e:
            logger.error(f"Section template application failed for {section}: {str(e)}")
            return f"Error processing {section.value} section"
    
    def _format_transcript(self, transcript_entries: List[Dict[str, Any]], 
                          format_options: FormatOptions) -> str:
        """Format transcript entries"""
        formatted_entries = []
        
        for entry in transcript_entries:
            parts = []
            
            # Add timestamp if enabled
            if format_options.include_timestamps and 'time' in entry:
                timestamp = format_options.timestamp_format.format(time=entry['time'])
                parts.append(timestamp)
            
            # Add speaker name if enabled
            if format_options.include_speaker_names and 'speaker' in entry:
                speaker = format_options.speaker_name_format.format(name=entry['speaker'])
                parts.append(speaker)
            
            # Add text
            if 'text' in entry:
                parts.append(entry['text'])
            
            # Combine parts
            entry_text = ' '.join(parts)
            
            # Apply word wrapping if enabled
            if format_options.word_wrap and format_options.max_line_length:
                entry_text = self._wrap_text(entry_text, format_options.max_line_length)
            
            formatted_entries.append(entry_text)
        
        return '\n\n'.join(formatted_entries)
    
    def _format_participants(self, participants: List[Dict[str, Any]], 
                           format_options: FormatOptions) -> str:
        """Format participant statistics"""
        formatted_participants = []
        
        for participant in participants:
            name = participant.get('name', 'Unknown')
            speaking_time = participant.get('speaking_time', '0 min')
            percentage = participant.get('percentage', 0)
            
            formatted_participants.append(f"{name}: {speaking_time} ({percentage}%)")
        
        return '\n'.join(formatted_participants)
    
    def _format_transcript_section(self, content: str, format_options: FormatOptions) -> str:
        """Apply transcript-specific formatting"""
        # Apply paragraph spacing
        if format_options.paragraph_spacing > 1:
            spacing = '\n' * (format_options.paragraph_spacing - 1)
            content = content.replace('\n\n', f'\n\n{spacing}')
        
        return content
    
    def _wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text to specified length"""
        import textwrap
        return textwrap.fill(text, width=max_length)
    
    def _apply_post_processing(self, content: str, format_options: FormatOptions) -> str:
        """Apply post-processing to formatted content"""
        # Remove excessive blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Ensure consistent line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip() + '\n'
    
    def save_template(self, template: ExportTemplate) -> bool:
        """Save template to disk"""
        try:
            template_data = self._serialize_template(template)
            file_path = self.templates_dir / f"{template.config.name}.json"
            
            with open(file_path, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            logger.info(f"Saved template: {template.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save template {template.config.name}: {str(e)}")
            return False
    
    def load_template(self, template_name: str) -> Optional[ExportTemplate]:
        """Load template from disk"""
        try:
            file_path = self.templates_dir / f"{template_name}.json"
            
            if not file_path.exists():
                logger.warning(f"Template file not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                template_data = json.load(f)
            
            template = self._deserialize_template(template_data)
            self.templates[template_name] = template
            
            logger.info(f"Loaded template: {template_name}")
            return template
            
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {str(e)}")
            return None
    
    def _serialize_template(self, template: ExportTemplate) -> Dict[str, Any]:
        """Serialize template to dictionary"""
        return {
            'config': asdict(template.config),
            'content_templates': {section.value: content for section, content in template.content_templates.items()},
            'variables': template.variables,
            'validation_rules': template.validation_rules
        }
    
    def _deserialize_template(self, template_data: Dict[str, Any]) -> ExportTemplate:
        """Deserialize template from dictionary"""
        config_data = template_data['config']
        
        # Convert enum strings back to enums
        config_data['template_type'] = TemplateType(config_data['template_type'])
        config_data['sections'] = [ContentSection(section) for section in config_data['sections']]
        
        # Create config and format options
        format_options_data = config_data.pop('format_options')
        format_options = FormatOptions(**format_options_data)
        config_data['format_options'] = format_options
        
        # Parse datetime
        if 'created_at' in config_data:
            config_data['created_at'] = datetime.fromisoformat(config_data['created_at'])
        
        config = TemplateConfig(**config_data)
        
        # Create content templates dictionary
        content_templates = {}
        for section_name, content in template_data.get('content_templates', {}).items():
            content_templates[ContentSection(section_name)] = content
        
        return ExportTemplate(
            config=config,
            content_templates=content_templates,
            variables=template_data.get('variables', {}),
            validation_rules=template_data.get('validation_rules', {})
        )
    
    def validate_template(self, template: ExportTemplate) -> List[str]:
        """Validate template configuration"""
        errors = []
        
        # Check required fields
        if not template.config.name:
            errors.append("Template name is required")
        
        if not template.config.sections:
            errors.append("Template must have at least one section")
        
        # Check content templates match sections
        for section in template.config.sections:
            if section not in template.content_templates:
                errors.append(f"Missing content template for section: {section.value}")
        
        # Validate template syntax
        for section, content in template.content_templates.items():
            try:
                Template(content)
            except Exception as e:
                errors.append(f"Invalid template syntax in {section.value}: {str(e)}")
        
        return errors
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get template information"""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        return {
            'name': template.config.name,
            'description': template.config.description,
            'type': template.config.template_type.value,
            'sections': [section.value for section in template.config.sections],
            'created_at': template.config.created_at.isoformat(),
            'version': template.config.version
        }

def create_format_customizer(templates_dir: str = "/tmp/silentSteno_templates") -> FormatCustomizer:
    """Create format customizer with specified templates directory"""
    return FormatCustomizer(templates_dir)

def create_export_template(name: str, template_type: TemplateType, 
                          sections: List[ContentSection]) -> ExportTemplate:
    """Create export template with basic configuration"""
    config = TemplateConfig(
        name=name,
        description=f"Custom {template_type.value} template",
        template_type=template_type,
        format_options=FormatOptions(),
        sections=sections
    )
    
    return ExportTemplate(config=config)

def customize_format(template_name: str, session_data: Dict[str, Any], 
                    customizer: Optional[FormatCustomizer] = None) -> str:
    """Convenience function to customize format"""
    if customizer is None:
        customizer = create_format_customizer()
    
    return customizer.customize_format(template_name, session_data)

def validate_template(template: ExportTemplate) -> List[str]:
    """Validate template configuration"""
    customizer = create_format_customizer()
    return customizer.validate_template(template)

def apply_template_settings(template: ExportTemplate, settings: Dict[str, Any]) -> ExportTemplate:
    """Apply settings to template"""
    # Update format options
    for key, value in settings.items():
        if hasattr(template.config.format_options, key):
            setattr(template.config.format_options, key, value)
    
    return template