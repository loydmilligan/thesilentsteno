#!/usr/bin/env python3
"""
Output Formatter Module

Multi-format structured output generation for LLM analysis results
with JSON, Markdown, HTML, and custom format support.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"


class TemplateType(Enum):
    """Template types for different content"""
    SUMMARY = "summary"
    ACTION_ITEMS = "action_items"
    TOPICS = "topics"
    PARTICIPANTS = "participants"
    FULL_REPORT = "full_report"
    CUSTOM = "custom"


@dataclass
class OutputConfig:
    """Configuration for output formatting"""
    
    # Format settings
    output_format: OutputFormat = OutputFormat.JSON
    template_type: TemplateType = TemplateType.FULL_REPORT
    
    # Styling settings
    include_metadata: bool = True
    include_timestamps: bool = True
    include_confidence_scores: bool = False
    
    # Content settings
    max_summary_length: int = 1000
    max_action_items: int = 20
    max_topics: int = 10
    
    # Formatting options
    indent_size: int = 2
    line_breaks: bool = True
    pretty_print: bool = True
    
    # Custom settings
    custom_template: Optional[str] = None
    custom_css: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "output_format": self.output_format.value,
            "template_type": self.template_type.value,
            "include_metadata": self.include_metadata,
            "include_timestamps": self.include_timestamps,
            "include_confidence_scores": self.include_confidence_scores,
            "max_summary_length": self.max_summary_length,
            "max_action_items": self.max_action_items,
            "max_topics": self.max_topics,
            "indent_size": self.indent_size,
            "line_breaks": self.line_breaks,
            "pretty_print": self.pretty_print,
            "custom_template": self.custom_template,
            "custom_css": self.custom_css,
            "custom_headers": self.custom_headers
        }


@dataclass
class FormattedOutput:
    """Formatted output result"""
    
    # Content
    content: str
    format_type: OutputFormat
    
    # Metadata
    generation_time: datetime = field(default_factory=datetime.now)
    content_length: int = 0
    
    # Quality metrics
    format_valid: bool = True
    formatting_errors: List[str] = field(default_factory=list)
    
    # Additional data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    template_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "format_type": self.format_type.value,
            "generation_time": self.generation_time.isoformat(),
            "content_length": self.content_length,
            "format_valid": self.format_valid,
            "formatting_errors": self.formatting_errors,
            "raw_data": self.raw_data,
            "template_used": self.template_used
        }


class OutputFormatter:
    """Main output formatter"""
    
    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        
        # Built-in templates
        self.templates = {
            OutputFormat.MARKDOWN: self._load_markdown_templates(),
            OutputFormat.HTML: self._load_html_templates(),
            OutputFormat.TEXT: self._load_text_templates()
        }
        
        logger.info(f"OutputFormatter initialized with {self.config.output_format.value} format")
        
    def _load_markdown_templates(self) -> Dict[TemplateType, str]:
        """Load Markdown templates"""
        return {
            TemplateType.SUMMARY: """# Meeting Summary

{metadata}

## Overview
{summary}

{additional_content}
""",
            TemplateType.ACTION_ITEMS: """# Action Items

{metadata}

{action_items}
""",
            TemplateType.TOPICS: """# Meeting Topics

{metadata}

{topics}
""",
            TemplateType.PARTICIPANTS: """# Participant Analysis

{metadata}

{participants}
""",
            TemplateType.FULL_REPORT: """# Meeting Analysis Report

{metadata}

## Summary
{summary}

## Action Items
{action_items}

## Topics Discussed
{topics}

## Participant Analysis
{participants}

{additional_sections}
"""
        }
        
    def _load_html_templates(self) -> Dict[TemplateType, str]:
        """Load HTML templates"""
        return {
            TemplateType.SUMMARY: """<!DOCTYPE html>
<html>
<head>
    <title>Meeting Summary</title>
    <style>{css}</style>
</head>
<body>
    <h1>Meeting Summary</h1>
    {metadata}
    <div class="summary">
        <h2>Overview</h2>
        {summary}
    </div>
    {additional_content}
</body>
</html>""",
            TemplateType.FULL_REPORT: """<!DOCTYPE html>
<html>
<head>
    <title>Meeting Analysis Report</title>
    <style>{css}</style>
</head>
<body>
    <h1>Meeting Analysis Report</h1>
    {metadata}
    
    <div class="summary">
        <h2>Summary</h2>
        {summary}
    </div>
    
    <div class="action-items">
        <h2>Action Items</h2>
        {action_items}
    </div>
    
    <div class="topics">
        <h2>Topics Discussed</h2>
        {topics}
    </div>
    
    <div class="participants">
        <h2>Participant Analysis</h2>
        {participants}
    </div>
    
    {additional_sections}
</body>
</html>"""
        }
        
    def _load_text_templates(self) -> Dict[TemplateType, str]:
        """Load text templates"""
        return {
            TemplateType.SUMMARY: """MEETING SUMMARY
===============

{metadata}

OVERVIEW
--------
{summary}

{additional_content}
""",
            TemplateType.FULL_REPORT: """MEETING ANALYSIS REPORT
=======================

{metadata}

SUMMARY
-------
{summary}

ACTION ITEMS
-----------
{action_items}

TOPICS DISCUSSED
---------------
{topics}

PARTICIPANT ANALYSIS
-------------------
{participants}

{additional_sections}
"""
        }
        
    def format_output(self, data: Dict[str, Any]) -> FormattedOutput:
        """Format data into specified output format"""
        try:
            start_time = datetime.now()
            
            # Prepare data
            formatted_data = self._prepare_data(data)
            
            # Generate content based on format
            if self.config.output_format == OutputFormat.JSON:
                content = self._format_json(formatted_data)
            elif self.config.output_format == OutputFormat.MARKDOWN:
                content = self._format_markdown(formatted_data)
            elif self.config.output_format == OutputFormat.HTML:
                content = self._format_html(formatted_data)
            elif self.config.output_format == OutputFormat.TEXT:
                content = self._format_text(formatted_data)
            elif self.config.output_format == OutputFormat.CSV:
                content = self._format_csv(formatted_data)
            elif self.config.output_format == OutputFormat.XML:
                content = self._format_xml(formatted_data)
            elif self.config.output_format == OutputFormat.YAML:
                content = self._format_yaml(formatted_data)
            else:
                raise ValueError(f"Unsupported format: {self.config.output_format}")
                
            # Validate format
            format_valid, errors = self._validate_format(content, self.config.output_format)
            
            # Create result
            result = FormattedOutput(
                content=content,
                format_type=self.config.output_format,
                generation_time=start_time,
                content_length=len(content),
                format_valid=format_valid,
                formatting_errors=errors,
                raw_data=formatted_data,
                template_used=self.config.template_type.value
            )
            
            logger.info(f"Output formatted successfully ({len(content)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            
            return FormattedOutput(
                content="",
                format_type=self.config.output_format,
                format_valid=False,
                formatting_errors=[str(e)]
            )
            
    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for formatting"""
        prepared = data.copy()
        
        # Add metadata if enabled
        if self.config.include_metadata:
            prepared["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "format": self.config.output_format.value,
                "template": self.config.template_type.value
            }
            
        # Truncate content if needed
        if "summary" in prepared and len(prepared["summary"]) > self.config.max_summary_length:
            prepared["summary"] = prepared["summary"][:self.config.max_summary_length] + "..."
            
        if "action_items" in prepared and len(prepared["action_items"]) > self.config.max_action_items:
            prepared["action_items"] = prepared["action_items"][:self.config.max_action_items]
            
        if "topics" in prepared and len(prepared["topics"]) > self.config.max_topics:
            prepared["topics"] = prepared["topics"][:self.config.max_topics]
            
        return prepared
        
    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format as JSON"""
        if self.config.pretty_print:
            return json.dumps(data, indent=self.config.indent_size, ensure_ascii=False)
        else:
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
            
    def _format_markdown(self, data: Dict[str, Any]) -> str:
        """Format as Markdown"""
        template = self.templates[OutputFormat.MARKDOWN].get(
            self.config.template_type, 
            self.templates[OutputFormat.MARKDOWN][TemplateType.FULL_REPORT]
        )
        
        # Prepare template variables
        variables = {
            "metadata": self._format_metadata_markdown(data.get("metadata", {})),
            "summary": data.get("summary", "No summary available"),
            "action_items": self._format_action_items_markdown(data.get("action_items", [])),
            "topics": self._format_topics_markdown(data.get("topics", [])),
            "participants": self._format_participants_markdown(data.get("participants", [])),
            "additional_content": "",
            "additional_sections": ""
        }
        
        return template.format(**variables)
        
    def _format_html(self, data: Dict[str, Any]) -> str:
        """Format as HTML"""
        template = self.templates[OutputFormat.HTML].get(
            self.config.template_type,
            self.templates[OutputFormat.HTML][TemplateType.FULL_REPORT]
        )
        
        # Prepare template variables
        variables = {
            "css": self._get_default_css(),
            "metadata": self._format_metadata_html(data.get("metadata", {})),
            "summary": self._html_escape(data.get("summary", "No summary available")),
            "action_items": self._format_action_items_html(data.get("action_items", [])),
            "topics": self._format_topics_html(data.get("topics", [])),
            "participants": self._format_participants_html(data.get("participants", [])),
            "additional_content": "",
            "additional_sections": ""
        }
        
        return template.format(**variables)
        
    def _format_text(self, data: Dict[str, Any]) -> str:
        """Format as plain text"""
        template = self.templates[OutputFormat.TEXT].get(
            self.config.template_type,
            self.templates[OutputFormat.TEXT][TemplateType.FULL_REPORT]
        )
        
        # Prepare template variables
        variables = {
            "metadata": self._format_metadata_text(data.get("metadata", {})),
            "summary": data.get("summary", "No summary available"),
            "action_items": self._format_action_items_text(data.get("action_items", [])),
            "topics": self._format_topics_text(data.get("topics", [])),
            "participants": self._format_participants_text(data.get("participants", [])),
            "additional_content": "",
            "additional_sections": ""
        }
        
        return template.format(**variables)
        
    def _format_csv(self, data: Dict[str, Any]) -> str:
        """Format as CSV"""
        csv_lines = []
        
        # Action items CSV
        if "action_items" in data and data["action_items"]:
            csv_lines.append("Type,Description,Assignee,Priority,Deadline,Status")
            for item in data["action_items"]:
                line = f"Action Item,\"{item.get('description', '')}\",{item.get('assignee', '')},{item.get('priority', '')},{item.get('deadline', '')},{item.get('status', '')}"
                csv_lines.append(line)
                
        # Topics CSV
        if "topics" in data and data["topics"]:
            csv_lines.append("\nType,Name,Importance,Relevance,Mentions")
            for topic in data["topics"]:
                line = f"Topic,\"{topic.get('name', '')}\",{topic.get('importance', '')},{topic.get('relevance_score', '')},{topic.get('mention_count', '')}"
                csv_lines.append(line)
                
        return "\n".join(csv_lines)
        
    def _format_xml(self, data: Dict[str, Any]) -> str:
        """Format as XML"""
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<meeting_analysis>')
        
        # Summary
        if "summary" in data:
            xml_lines.append(f'  <summary>{self._xml_escape(data["summary"])}</summary>')
            
        # Action items
        if "action_items" in data and data["action_items"]:
            xml_lines.append('  <action_items>')
            for item in data["action_items"]:
                xml_lines.append('    <item>')
                xml_lines.append(f'      <description>{self._xml_escape(item.get("description", ""))}</description>')
                xml_lines.append(f'      <assignee>{self._xml_escape(item.get("assignee", ""))}</assignee>')
                xml_lines.append(f'      <priority>{item.get("priority", "")}</priority>')
                xml_lines.append('    </item>')
            xml_lines.append('  </action_items>')
            
        # Topics
        if "topics" in data and data["topics"]:
            xml_lines.append('  <topics>')
            for topic in data["topics"]:
                xml_lines.append('    <topic>')
                xml_lines.append(f'      <name>{self._xml_escape(topic.get("name", ""))}</name>')
                xml_lines.append(f'      <importance>{topic.get("importance", "")}</importance>')
                xml_lines.append(f'      <relevance>{topic.get("relevance_score", "")}</relevance>')
                xml_lines.append('    </topic>')
            xml_lines.append('  </topics>')
            
        xml_lines.append('</meeting_analysis>')
        return '\n'.join(xml_lines)
        
    def _format_yaml(self, data: Dict[str, Any]) -> str:
        """Format as YAML"""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, indent=self.config.indent_size)
        except ImportError:
            logger.warning("PyYAML not available, using simple YAML format")
            return self._format_simple_yaml(data)
            
    def _format_simple_yaml(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Simple YAML formatting without PyYAML"""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_simple_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.append(self._format_simple_yaml(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
                
        return "\n".join(lines)
        
    def _format_metadata_markdown(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for Markdown"""
        if not metadata:
            return ""
            
        lines = []
        if self.config.include_timestamps:
            lines.append(f"*Generated: {metadata.get('generated_at', 'Unknown')}*")
        if self.config.include_metadata:
            lines.append(f"*Format: {metadata.get('format', 'Unknown')}*")
            
        return "\n".join(lines)
        
    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for HTML"""
        if not metadata:
            return ""
            
        lines = ['<div class="metadata">']
        if self.config.include_timestamps:
            lines.append(f'<p><em>Generated: {metadata.get("generated_at", "Unknown")}</em></p>')
        if self.config.include_metadata:
            lines.append(f'<p><em>Format: {metadata.get("format", "Unknown")}</em></p>')
        lines.append('</div>')
        
        return "\n".join(lines)
        
    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for text"""
        if not metadata:
            return ""
            
        lines = []
        if self.config.include_timestamps:
            lines.append(f"Generated: {metadata.get('generated_at', 'Unknown')}")
        if self.config.include_metadata:
            lines.append(f"Format: {metadata.get('format', 'Unknown')}")
            
        return "\n".join(lines)
        
    def _format_action_items_markdown(self, items: List[Dict[str, Any]]) -> str:
        """Format action items for Markdown"""
        if not items:
            return "No action items found."
            
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. **{item.get('description', 'Unknown')}**")
            lines.append(f"   - Assignee: {item.get('assignee', 'Unknown')}")
            lines.append(f"   - Priority: {item.get('priority', 'Unknown')}")
            if item.get('deadline'):
                lines.append(f"   - Deadline: {item.get('deadline')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_action_items_html(self, items: List[Dict[str, Any]]) -> str:
        """Format action items for HTML"""
        if not items:
            return "<p>No action items found.</p>"
            
        lines = ['<ol>']
        for item in items:
            lines.append('<li>')
            lines.append(f'<strong>{self._html_escape(item.get("description", "Unknown"))}</strong>')
            lines.append('<ul>')
            lines.append(f'<li>Assignee: {self._html_escape(item.get("assignee", "Unknown"))}</li>')
            lines.append(f'<li>Priority: {item.get("priority", "Unknown")}</li>')
            if item.get('deadline'):
                lines.append(f'<li>Deadline: {item.get("deadline")}</li>')
            lines.append('</ul>')
            lines.append('</li>')
        lines.append('</ol>')
        
        return "\n".join(lines)
        
    def _format_action_items_text(self, items: List[Dict[str, Any]]) -> str:
        """Format action items for text"""
        if not items:
            return "No action items found."
            
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.get('description', 'Unknown')}")
            lines.append(f"   Assignee: {item.get('assignee', 'Unknown')}")
            lines.append(f"   Priority: {item.get('priority', 'Unknown')}")
            if item.get('deadline'):
                lines.append(f"   Deadline: {item.get('deadline')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_topics_markdown(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for Markdown"""
        if not topics:
            return "No topics identified."
            
        lines = []
        for i, topic in enumerate(topics, 1):
            lines.append(f"{i}. **{topic.get('name', 'Unknown')}**")
            lines.append(f"   - Importance: {topic.get('importance', 'Unknown')}")
            lines.append(f"   - Relevance: {topic.get('relevance_score', 'Unknown')}")
            lines.append(f"   - Mentions: {topic.get('mention_count', 'Unknown')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_topics_html(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for HTML"""
        if not topics:
            return "<p>No topics identified.</p>"
            
        lines = ['<ol>']
        for topic in topics:
            lines.append('<li>')
            lines.append(f'<strong>{self._html_escape(topic.get("name", "Unknown"))}</strong>')
            lines.append('<ul>')
            lines.append(f'<li>Importance: {topic.get("importance", "Unknown")}</li>')
            lines.append(f'<li>Relevance: {topic.get("relevance_score", "Unknown")}</li>')
            lines.append(f'<li>Mentions: {topic.get("mention_count", "Unknown")}</li>')
            lines.append('</ul>')
            lines.append('</li>')
        lines.append('</ol>')
        
        return "\n".join(lines)
        
    def _format_topics_text(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for text"""
        if not topics:
            return "No topics identified."
            
        lines = []
        for i, topic in enumerate(topics, 1):
            lines.append(f"{i}. {topic.get('name', 'Unknown')}")
            lines.append(f"   Importance: {topic.get('importance', 'Unknown')}")
            lines.append(f"   Relevance: {topic.get('relevance_score', 'Unknown')}")
            lines.append(f"   Mentions: {topic.get('mention_count', 'Unknown')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_participants_markdown(self, participants: List[Dict[str, Any]]) -> str:
        """Format participants for Markdown"""
        if not participants:
            return "No participant analysis available."
            
        lines = []
        for participant in participants:
            lines.append(f"### {participant.get('name', 'Unknown')}")
            lines.append(f"- Participation: {participant.get('participation_level', 'Unknown')}")
            lines.append(f"- Speaking time: {participant.get('speaking_time', 'Unknown')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_participants_html(self, participants: List[Dict[str, Any]]) -> str:
        """Format participants for HTML"""
        if not participants:
            return "<p>No participant analysis available.</p>"
            
        lines = []
        for participant in participants:
            lines.append(f'<h3>{self._html_escape(participant.get("name", "Unknown"))}</h3>')
            lines.append('<ul>')
            lines.append(f'<li>Participation: {participant.get("participation_level", "Unknown")}</li>')
            lines.append(f'<li>Speaking time: {participant.get("speaking_time", "Unknown")}</li>')
            lines.append('</ul>')
            
        return "\n".join(lines)
        
    def _format_participants_text(self, participants: List[Dict[str, Any]]) -> str:
        """Format participants for text"""
        if not participants:
            return "No participant analysis available."
            
        lines = []
        for participant in participants:
            lines.append(f"{participant.get('name', 'Unknown')}:")
            lines.append(f"  Participation: {participant.get('participation_level', 'Unknown')}")
            lines.append(f"  Speaking time: {participant.get('speaking_time', 'Unknown')}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _get_default_css(self) -> str:
        """Get default CSS for HTML output"""
        if self.config.custom_css:
            return self.config.custom_css
            
        return """
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; border-bottom: 2px solid #333; }
        h2 { color: #666; border-bottom: 1px solid #ccc; }
        h3 { color: #888; }
        .metadata { font-style: italic; color: #666; margin-bottom: 20px; }
        .summary { margin-bottom: 30px; }
        .action-items { margin-bottom: 30px; }
        .topics { margin-bottom: 30px; }
        .participants { margin-bottom: 30px; }
        ul { margin: 10px 0; }
        li { margin: 5px 0; }
        """
        
    def _html_escape(self, text: str) -> str:
        """Escape HTML special characters"""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        
    def _xml_escape(self, text: str) -> str:
        """Escape XML special characters"""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
        
    def _validate_format(self, content: str, format_type: OutputFormat) -> Tuple[bool, List[str]]:
        """Validate formatted content"""
        errors = []
        
        try:
            if format_type == OutputFormat.JSON:
                json.loads(content)
            elif format_type == OutputFormat.HTML:
                # Basic HTML validation
                if not content.strip().startswith('<!DOCTYPE'):
                    errors.append("HTML document should start with DOCTYPE")
                if '<html>' not in content or '</html>' not in content:
                    errors.append("HTML document should have html tags")
            elif format_type == OutputFormat.XML:
                # Basic XML validation
                if not content.strip().startswith('<?xml'):
                    errors.append("XML document should start with declaration")
                    
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return len(errors) == 0, errors
        
    def save_to_file(self, output: FormattedOutput, filepath: str) -> bool:
        """Save formatted output to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(output.content)
            logger.info(f"Output saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get formatter status"""
        return {
            "config": self.config.to_dict(),
            "supported_formats": [fmt.value for fmt in OutputFormat],
            "template_types": [ttype.value for ttype in TemplateType],
            "templates_loaded": sum(len(templates) for templates in self.templates.values())
        }


# Factory functions
def create_json_formatter() -> OutputFormatter:
    """Create JSON formatter"""
    config = OutputConfig(
        output_format=OutputFormat.JSON,
        pretty_print=True,
        include_metadata=True,
        include_timestamps=True
    )
    return OutputFormatter(config)


def create_markdown_formatter() -> OutputFormatter:
    """Create Markdown formatter"""
    config = OutputConfig(
        output_format=OutputFormat.MARKDOWN,
        template_type=TemplateType.FULL_REPORT,
        include_metadata=True,
        include_timestamps=True
    )
    return OutputFormatter(config)


def create_html_formatter() -> OutputFormatter:
    """Create HTML formatter"""
    config = OutputConfig(
        output_format=OutputFormat.HTML,
        template_type=TemplateType.FULL_REPORT,
        include_metadata=True,
        include_timestamps=True,
        pretty_print=True
    )
    return OutputFormatter(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Output Formatter Test")
    parser.add_argument("--format", type=str, default="json",
                       choices=["json", "markdown", "html", "text", "csv"],
                       help="Output format")
    parser.add_argument("--template", type=str, default="full_report",
                       choices=["summary", "action_items", "topics", "full_report"],
                       help="Template type")
    args = parser.parse_args()
    
    # Create formatter
    config = OutputConfig(
        output_format=OutputFormat(args.format),
        template_type=TemplateType(args.template),
        pretty_print=True,
        include_metadata=True
    )
    formatter = OutputFormatter(config)
    
    # Sample data
    sample_data = {
        "summary": "This was a productive meeting about project planning and resource allocation.",
        "action_items": [
            {"description": "Review project timeline", "assignee": "John", "priority": "high", "deadline": "2024-07-20"},
            {"description": "Prepare budget proposal", "assignee": "Sarah", "priority": "medium", "deadline": "2024-07-25"}
        ],
        "topics": [
            {"name": "Project Timeline", "importance": "high", "relevance_score": 0.8, "mention_count": 5},
            {"name": "Budget Planning", "importance": "medium", "relevance_score": 0.6, "mention_count": 3}
        ],
        "participants": [
            {"name": "John", "participation_level": "high", "speaking_time": "35%"},
            {"name": "Sarah", "participation_level": "medium", "speaking_time": "25%"}
        ]
    }
    
    try:
        print(f"Formatter status: {formatter.get_status()}")
        
        # Format output
        print(f"Formatting output as {args.format}...")
        result = formatter.format_output(sample_data)
        
        if result.format_valid:
            print("Formatting completed successfully!")
            print(f"Content length: {result.content_length}")
            print(f"Generated at: {result.generation_time}")
            
            print("\nFormatted output:")
            print("-" * 50)
            print(result.content)
            
        else:
            print("Formatting failed!")
            print(f"Errors: {result.formatting_errors}")
            
    except Exception as e:
        print(f"Error: {e}")