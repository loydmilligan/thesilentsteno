#!/usr/bin/env python3
"""
Prompt Templates Module

Comprehensive prompt template system for meeting analysis tasks including
summarization, action item extraction, and topic identification.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of prompt templates"""
    SUMMARIZATION = "summarization"
    ACTION_ITEMS = "action_items"
    TOPIC_IDENTIFICATION = "topic_identification"
    PARTICIPANT_ANALYSIS = "participant_analysis"
    MEETING_INSIGHTS = "meeting_insights"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DECISION_TRACKING = "decision_tracking"
    FOLLOW_UP = "follow_up"


class MeetingType(Enum):
    """Types of meetings for context-aware templates"""
    GENERAL = "general"
    STANDUP = "standup"
    PLANNING = "planning"
    RETROSPECTIVE = "retrospective"
    REVIEW = "review"
    INTERVIEW = "interview"
    PRESENTATION = "presentation"
    BRAINSTORMING = "brainstorming"


class OutputFormat(Enum):
    """Output format for template responses"""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    BULLET_POINTS = "bullet_points"
    STRUCTURED = "structured"


@dataclass
class TemplateConfig:
    """Configuration for prompt templates"""
    
    # Template settings
    template_type: TemplateType = TemplateType.SUMMARIZATION
    meeting_type: MeetingType = MeetingType.GENERAL
    output_format: OutputFormat = OutputFormat.TEXT
    
    # Context settings
    include_timestamps: bool = True
    include_speakers: bool = True
    include_confidence: bool = False
    
    # Content settings
    max_summary_length: int = 500
    max_action_items: int = 10
    max_topics: int = 5
    
    # Language settings
    language: str = "en"
    tone: str = "professional"
    
    # Customization
    custom_instructions: str = ""
    context_information: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "template_type": self.template_type.value,
            "meeting_type": self.meeting_type.value,
            "output_format": self.output_format.value,
            "include_timestamps": self.include_timestamps,
            "include_speakers": self.include_speakers,
            "include_confidence": self.include_confidence,
            "max_summary_length": self.max_summary_length,
            "max_action_items": self.max_action_items,
            "max_topics": self.max_topics,
            "language": self.language,
            "tone": self.tone,
            "custom_instructions": self.custom_instructions,
            "context_information": self.context_information
        }


@dataclass
class PromptTemplate:
    """A prompt template with metadata"""
    
    # Template identification
    template_id: str
    name: str
    description: str
    
    # Template content
    system_prompt: str
    user_prompt_template: str
    
    # Template metadata
    template_type: TemplateType
    meeting_type: MeetingType
    output_format: OutputFormat
    
    # Template variables
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    
    # Example usage
    example_input: str = ""
    example_output: str = ""
    
    # Template settings
    max_tokens: int = 512
    temperature: float = 0.7
    
    def format_prompt(self, variables: Dict[str, Any]) -> str:
        """Format the prompt template with variables"""
        try:
            # Check required variables
            missing_vars = [var for var in self.required_variables if var not in variables]
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")
                
            # Format template
            formatted_prompt = self.user_prompt_template.format(**variables)
            
            # Combine system and user prompts
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{formatted_prompt}"
            else:
                full_prompt = formatted_prompt
                
            return full_prompt
            
        except Exception as e:
            logger.error(f"Failed to format prompt template {self.template_id}: {e}")
            raise
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "template_type": self.template_type.value,
            "meeting_type": self.meeting_type.value,
            "output_format": self.output_format.value,
            "required_variables": self.required_variables,
            "optional_variables": self.optional_variables,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class PromptTemplateManager:
    """Manager for prompt templates"""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        self.config = config or TemplateConfig()
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Load built-in templates
        self._load_builtin_templates()
        
        logger.info(f"PromptTemplateManager initialized with {len(self.templates)} templates")
        
    def _load_builtin_templates(self):
        """Load built-in prompt templates"""
        
        # Meeting Summarization Template
        self.templates["meeting_summary"] = PromptTemplate(
            template_id="meeting_summary",
            name="Meeting Summary",
            description="Comprehensive meeting summarization with key points and decisions",
            system_prompt="""You are an expert meeting analyst. Your task is to create comprehensive, accurate summaries of meeting transcripts. Focus on:
- Key discussion points and decisions made
- Important information shared
- Action items and next steps
- Participant contributions
- Meeting outcomes and conclusions

Be concise, objective, and professional. Maintain the original context and meaning.""",
            user_prompt_template="""Please analyze this meeting transcript and provide a comprehensive summary:

MEETING TRANSCRIPT:
{transcript}

{context_info}

Please provide a summary that includes:
1. Meeting overview and main topics discussed
2. Key decisions and outcomes
3. Important information shared
4. Next steps and action items
5. Notable participant contributions

Format: {output_format}
Maximum length: {max_length} words
{custom_instructions}""",
            template_type=TemplateType.SUMMARIZATION,
            meeting_type=MeetingType.GENERAL,
            output_format=OutputFormat.TEXT,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "max_length", "custom_instructions"],
            example_input="Meeting transcript discussing project updates...",
            example_output="Meeting Summary: The team discussed project progress...",
            max_tokens=1024,
            temperature=0.6
        )
        
        # Action Items Template
        self.templates["action_items"] = PromptTemplate(
            template_id="action_items",
            name="Action Items Extraction",
            description="Extract action items with assignees and deadlines from meeting transcripts",
            system_prompt="""You are an expert at identifying and extracting action items from meeting discussions. Your task is to:
- Identify clear action items and tasks
- Determine who is responsible (assignee)
- Extract deadlines or timeframes when mentioned
- Prioritize items based on urgency and importance
- Format items clearly and concisely

Focus on explicit commitments and tasks, not general discussions.""",
            user_prompt_template="""Please analyze this meeting transcript and extract all action items:

MEETING TRANSCRIPT:
{transcript}

{context_info}

For each action item, provide:
- Action description (what needs to be done)
- Assignee (who is responsible)
- Deadline or timeframe (if mentioned)
- Priority level (High/Medium/Low)
- Context or background information

Format as {output_format}
Maximum {max_items} items
{custom_instructions}""",
            template_type=TemplateType.ACTION_ITEMS,
            meeting_type=MeetingType.GENERAL,
            output_format=OutputFormat.STRUCTURED,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "max_items", "custom_instructions"],
            example_input="Meeting transcript with task assignments...",
            example_output="Action Items: 1. John to review proposal by Friday...",
            max_tokens=512,
            temperature=0.5
        )
        
        # Topic Identification Template
        self.templates["topic_identification"] = PromptTemplate(
            template_id="topic_identification",
            name="Topic Identification",
            description="Identify key topics and themes discussed in meetings",
            system_prompt="""You are an expert at identifying and categorizing discussion topics from meeting transcripts. Your task is to:
- Identify main topics and themes discussed
- Categorize topics by importance and time spent
- Group related discussions together
- Provide context for each topic
- Highlight emerging themes and patterns

Focus on substantial discussions, not brief mentions.""",
            user_prompt_template="""Please analyze this meeting transcript and identify the main topics discussed:

MEETING TRANSCRIPT:
{transcript}

{context_info}

For each topic, provide:
- Topic name and description
- Key points discussed
- Time spent on topic (relative importance)
- Participants most involved
- Outcomes or decisions related to topic

Format as {output_format}
Maximum {max_topics} topics
{custom_instructions}""",
            template_type=TemplateType.TOPIC_IDENTIFICATION,
            meeting_type=MeetingType.GENERAL,
            output_format=OutputFormat.STRUCTURED,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "max_topics", "custom_instructions"],
            example_input="Meeting transcript with various discussion topics...",
            example_output="Topics: 1. Project Timeline - Discussion about delays...",
            max_tokens=512,
            temperature=0.6
        )
        
        # Participant Analysis Template
        self.templates["participant_analysis"] = PromptTemplate(
            template_id="participant_analysis",
            name="Participant Analysis",
            description="Analyze participant contributions and engagement in meetings",
            system_prompt="""You are an expert at analyzing participant behavior and contributions in meetings. Your task is to:
- Assess each participant's level of engagement
- Identify key contributors and their areas of expertise
- Analyze communication patterns and dynamics
- Highlight collaborative interactions
- Provide insights into meeting effectiveness

Be objective and professional in your analysis.""",
            user_prompt_template="""Please analyze participant contributions in this meeting transcript:

MEETING TRANSCRIPT:
{transcript}

{context_info}

For each participant, provide:
- Name and role (if mentioned)
- Level of participation (High/Medium/Low)
- Key contributions and expertise areas
- Communication style and approach
- Interaction patterns with other participants

Also provide:
- Overall meeting dynamics
- Collaboration effectiveness
- Areas for improvement

Format as {output_format}
{custom_instructions}""",
            template_type=TemplateType.PARTICIPANT_ANALYSIS,
            meeting_type=MeetingType.GENERAL,
            output_format=OutputFormat.STRUCTURED,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "custom_instructions"],
            example_input="Meeting transcript with speaker labels...",
            example_output="Participant Analysis: John - High participation, led discussion on...",
            max_tokens=768,
            temperature=0.7
        )
        
        # Standup Meeting Template
        self.templates["standup_summary"] = PromptTemplate(
            template_id="standup_summary",
            name="Standup Meeting Summary",
            description="Specialized summary for daily standup meetings",
            system_prompt="""You are specialized in analyzing daily standup meetings. Focus on:
- What each team member accomplished
- Current work in progress
- Blockers and challenges
- Team coordination and dependencies
- Progress toward sprint goals

Be concise and action-oriented.""",
            user_prompt_template="""Please analyze this daily standup meeting transcript:

MEETING TRANSCRIPT:
{transcript}

{context_info}

Provide a standup summary including:
- Team member updates (completed, in progress, blockers)
- Overall team progress
- Identified blockers and impediments
- Cross-team dependencies
- Action items for resolution

Format as {output_format}
{custom_instructions}""",
            template_type=TemplateType.SUMMARIZATION,
            meeting_type=MeetingType.STANDUP,
            output_format=OutputFormat.STRUCTURED,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "custom_instructions"],
            example_input="Standup transcript with team updates...",
            example_output="Standup Summary: Sarah completed feature A, working on B...",
            max_tokens=512,
            temperature=0.5
        )
        
        # Meeting Insights Template
        self.templates["meeting_insights"] = PromptTemplate(
            template_id="meeting_insights",
            name="Meeting Insights",
            description="Extract deeper insights and patterns from meeting discussions",
            system_prompt="""You are an expert at extracting strategic insights from meeting discussions. Your task is to:
- Identify underlying patterns and trends
- Highlight strategic implications
- Assess decision-making effectiveness
- Identify potential risks or opportunities
- Provide recommendations for improvement

Focus on actionable insights that can improve future meetings and outcomes.""",
            user_prompt_template="""Please analyze this meeting transcript and provide strategic insights:

MEETING TRANSCRIPT:
{transcript}

{context_info}

Provide insights covering:
- Key patterns and trends observed
- Strategic implications of decisions
- Team dynamics and collaboration effectiveness
- Potential risks or opportunities identified
- Recommendations for improvement
- Follow-up suggestions

Format as {output_format}
{custom_instructions}""",
            template_type=TemplateType.MEETING_INSIGHTS,
            meeting_type=MeetingType.GENERAL,
            output_format=OutputFormat.STRUCTURED,
            required_variables=["transcript"],
            optional_variables=["context_info", "output_format", "custom_instructions"],
            example_input="Meeting transcript with strategic discussions...",
            example_output="Meeting Insights: The team showed strong alignment on...",
            max_tokens=768,
            temperature=0.7
        )
        
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
        
    def list_templates(self, template_type: Optional[TemplateType] = None,
                      meeting_type: Optional[MeetingType] = None) -> List[PromptTemplate]:
        """List templates with optional filtering"""
        templates = list(self.templates.values())
        
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
            
        if meeting_type:
            templates = [t for t in templates if t.meeting_type == meeting_type]
            
        return templates
        
    def add_template(self, template: PromptTemplate) -> bool:
        """Add a custom template"""
        try:
            self.templates[template.template_id] = template
            logger.info(f"Added template: {template.template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add template: {e}")
            return False
            
    def remove_template(self, template_id: str) -> bool:
        """Remove a template"""
        try:
            if template_id in self.templates:
                del self.templates[template_id]
                logger.info(f"Removed template: {template_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove template: {e}")
            return False
            
    def generate_prompt(self, template_id: str, variables: Dict[str, Any]) -> str:
        """Generate a prompt from template and variables"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
            
        # Add default variables
        default_vars = {
            "output_format": self.config.output_format.value,
            "max_length": self.config.max_summary_length,
            "max_items": self.config.max_action_items,
            "max_topics": self.config.max_topics,
            "custom_instructions": self.config.custom_instructions,
            "context_info": self.config.context_information
        }
        
        # Merge variables
        merged_vars = {**default_vars, **variables}
        
        return template.format_prompt(merged_vars)
        
    def get_recommended_template(self, meeting_type: MeetingType,
                               analysis_type: TemplateType) -> Optional[PromptTemplate]:
        """Get recommended template for meeting and analysis type"""
        templates = self.list_templates(analysis_type, meeting_type)
        
        if templates:
            return templates[0]
            
        # Fallback to general templates
        templates = self.list_templates(analysis_type, MeetingType.GENERAL)
        return templates[0] if templates else None
        
    def get_template_suggestions(self, transcript: str) -> List[str]:
        """Get template suggestions based on transcript content"""
        suggestions = []
        
        # Simple keyword-based suggestions
        transcript_lower = transcript.lower()
        
        if any(word in transcript_lower for word in ["action", "task", "todo", "assign", "deadline"]):
            suggestions.append("action_items")
            
        if any(word in transcript_lower for word in ["standup", "scrum", "sprint", "blocker"]):
            suggestions.append("standup_summary")
            
        if any(word in transcript_lower for word in ["decision", "vote", "agree", "conclusion"]):
            suggestions.append("meeting_insights")
            
        # Always suggest basic summary
        suggestions.append("meeting_summary")
        
        return suggestions
        
    def save_templates(self, filepath: str) -> bool:
        """Save templates to file"""
        try:
            templates_data = {
                template_id: template.to_dict() 
                for template_id, template in self.templates.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(templates_data, f, indent=2)
                
            logger.info(f"Saved {len(self.templates)} templates to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            return False
            
    def load_templates(self, filepath: str) -> bool:
        """Load templates from file"""
        try:
            with open(filepath, 'r') as f:
                templates_data = json.load(f)
                
            for template_id, template_data in templates_data.items():
                template = PromptTemplate(
                    template_id=template_data["template_id"],
                    name=template_data["name"],
                    description=template_data["description"],
                    system_prompt=template_data["system_prompt"],
                    user_prompt_template=template_data["user_prompt_template"],
                    template_type=TemplateType(template_data["template_type"]),
                    meeting_type=MeetingType(template_data["meeting_type"]),
                    output_format=OutputFormat(template_data["output_format"]),
                    required_variables=template_data["required_variables"],
                    optional_variables=template_data["optional_variables"],
                    example_input=template_data["example_input"],
                    example_output=template_data["example_output"],
                    max_tokens=template_data["max_tokens"],
                    temperature=template_data["temperature"]
                )
                self.templates[template_id] = template
                
            logger.info(f"Loaded {len(templates_data)} templates from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get template manager status"""
        return {
            "total_templates": len(self.templates),
            "templates_by_type": {
                template_type.value: len([t for t in self.templates.values() 
                                        if t.template_type == template_type])
                for template_type in TemplateType
            },
            "templates_by_meeting_type": {
                meeting_type.value: len([t for t in self.templates.values() 
                                       if t.meeting_type == meeting_type])
                for meeting_type in MeetingType
            },
            "config": self.config.to_dict()
        }


# Factory functions
def create_meeting_templates() -> PromptTemplateManager:
    """Create template manager for general meetings"""
    config = TemplateConfig(
        meeting_type=MeetingType.GENERAL,
        output_format=OutputFormat.STRUCTURED,
        include_timestamps=True,
        include_speakers=True,
        tone="professional"
    )
    return PromptTemplateManager(config)


def create_analysis_templates() -> PromptTemplateManager:
    """Create template manager for detailed analysis"""
    config = TemplateConfig(
        meeting_type=MeetingType.GENERAL,
        output_format=OutputFormat.STRUCTURED,
        include_timestamps=True,
        include_speakers=True,
        max_summary_length=1000,
        max_action_items=15,
        max_topics=8,
        tone="analytical"
    )
    return PromptTemplateManager(config)


def create_standup_templates() -> PromptTemplateManager:
    """Create template manager for standup meetings"""
    config = TemplateConfig(
        meeting_type=MeetingType.STANDUP,
        output_format=OutputFormat.BULLET_POINTS,
        include_timestamps=False,
        include_speakers=True,
        max_summary_length=300,
        max_action_items=5,
        tone="concise"
    )
    return PromptTemplateManager(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Template Manager Test")
    parser.add_argument("--template", type=str, default="meeting_summary", 
                       help="Template to test")
    parser.add_argument("--transcript", type=str, required=True, 
                       help="Meeting transcript text")
    parser.add_argument("--output-format", type=str, default="text",
                       choices=["text", "json", "markdown", "structured"],
                       help="Output format")
    args = parser.parse_args()
    
    # Create template manager
    manager = create_meeting_templates()
    
    # Update config
    manager.config.output_format = OutputFormat(args.output_format)
    
    try:
        print(f"Template manager status: {manager.get_status()}")
        
        # Generate prompt
        variables = {
            "transcript": args.transcript,
            "output_format": args.output_format
        }
        
        prompt = manager.generate_prompt(args.template, variables)
        print(f"\nGenerated prompt for template '{args.template}':")
        print("-" * 50)
        print(prompt)
        
        # Show template suggestions
        suggestions = manager.get_template_suggestions(args.transcript)
        print(f"\nTemplate suggestions: {suggestions}")
        
        # List available templates
        templates = manager.list_templates()
        print(f"\nAvailable templates ({len(templates)}):")
        for template in templates:
            print(f"  - {template.template_id}: {template.name}")
            
    except Exception as e:
        print(f"Error: {e}")