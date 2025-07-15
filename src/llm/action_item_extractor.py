#!/usr/bin/env python3
"""
Action Item Extractor Module

Intelligent action item extraction with assignee identification, priority scoring,
and deadline detection from meeting transcripts.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import re
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Priority(Enum):
    """Action item priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Status(Enum):
    """Action item status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ActionItem:
    """Represents an action item"""
    
    # Core information
    id: str
    description: str
    assignee: str
    
    # Priority and timing
    priority: Priority = Priority.MEDIUM
    deadline: Optional[datetime] = None
    estimated_effort: Optional[str] = None
    
    # Status tracking
    status: Status = Status.OPEN
    created_date: datetime = field(default_factory=datetime.now)
    
    # Context
    context: str = ""
    meeting_reference: str = ""
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    source_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "description": self.description,
            "assignee": self.assignee,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_effort": self.estimated_effort,
            "status": self.status.value,
            "created_date": self.created_date.isoformat(),
            "context": self.context,
            "meeting_reference": self.meeting_reference,
            "dependencies": self.dependencies,
            "confidence": self.confidence,
            "source_text": self.source_text
        }


@dataclass
class ExtractorConfig:
    """Configuration for action item extraction"""
    
    # Extraction settings
    max_items: int = 10
    min_confidence: float = 0.6
    include_implied_actions: bool = True
    
    # Analysis settings
    detect_deadlines: bool = True
    detect_priorities: bool = True
    detect_dependencies: bool = True
    
    # Context settings
    context_window: int = 100  # Characters around action item
    include_speaker_context: bool = True
    
    # Output settings
    sort_by_priority: bool = True
    group_by_assignee: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_items": self.max_items,
            "min_confidence": self.min_confidence,
            "include_implied_actions": self.include_implied_actions,
            "detect_deadlines": self.detect_deadlines,
            "detect_priorities": self.detect_priorities,
            "detect_dependencies": self.detect_dependencies,
            "context_window": self.context_window,
            "include_speaker_context": self.include_speaker_context,
            "sort_by_priority": self.sort_by_priority,
            "group_by_assignee": self.group_by_assignee
        }


@dataclass
class ExtractorResult:
    """Result from action item extraction"""
    
    # Extracted items
    action_items: List[ActionItem] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    items_found: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    
    # Analysis results
    assignees_identified: List[str] = field(default_factory=list)
    deadlines_found: int = 0
    priorities_assigned: int = 0
    
    # Status
    success: bool = True
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_items": [item.to_dict() for item in self.action_items],
            "processing_time": self.processing_time,
            "items_found": self.items_found,
            "confidence_scores": self.confidence_scores,
            "assignees_identified": self.assignees_identified,
            "deadlines_found": self.deadlines_found,
            "priorities_assigned": self.priorities_assigned,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class ActionItemExtractor:
    """Intelligent action item extractor"""
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        
        # Action patterns
        self.action_patterns = [
            r"(?i)\b(will|going to|plan to|need to|should|must|have to)\s+([^.!?]+)",
            r"(?i)\b(action|task|todo|follow up|next step):\s*([^.!?]+)",
            r"(?i)\b(\w+)\s+(will|should|needs to|has to)\s+([^.!?]+)",
            r"(?i)\b(assigned to|responsibility of|owned by)\s+(\w+)",
            r"(?i)\b(deadline|due|by)\s+([^.!?]+)",
        ]
        
        # Priority indicators
        self.priority_indicators = {
            Priority.URGENT: ["urgent", "asap", "immediately", "critical", "emergency"],
            Priority.HIGH: ["high priority", "important", "quickly", "soon", "priority"],
            Priority.MEDIUM: ["medium", "normal", "regular", "standard"],
            Priority.LOW: ["low priority", "when possible", "eventually", "nice to have"]
        }
        
        # Deadline patterns
        self.deadline_patterns = [
            r"(?i)by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(?i)by\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})",
            r"(?i)by\s+(\d{1,2})/(\d{1,2})",
            r"(?i)by\s+(\d{1,2})-(\d{1,2})-(\d{4})",
            r"(?i)(deadline|due)\s+([^.!?]+)",
            r"(?i)within\s+(\d+)\s+(days|weeks|months)",
            r"(?i)in\s+(\d+)\s+(days|weeks|months)"
        ]
        
        # Common names for assignee detection
        self.common_names = set([
            "john", "jane", "mike", "sarah", "david", "lisa", "tom", "mary",
            "alex", "chris", "anna", "paul", "emma", "james", "kate", "mark"
        ])
        
        logger.info("ActionItemExtractor initialized")
        
    def extract_action_items(self, transcript: str, 
                           meeting_id: Optional[str] = None) -> ExtractorResult:
        """Extract action items from meeting transcript"""
        start_time = datetime.now()
        
        try:
            result = ExtractorResult()
            
            # Extract potential action items
            potential_items = self._find_action_patterns(transcript)
            
            # Process each potential item
            for item_text, context in potential_items:
                action_item = self._process_action_item(item_text, context, meeting_id)
                
                if action_item and action_item.confidence >= self.config.min_confidence:
                    result.action_items.append(action_item)
                    result.confidence_scores.append(action_item.confidence)
                    
            # Sort and limit results
            if self.config.sort_by_priority:
                result.action_items.sort(key=lambda x: self._priority_score(x.priority), reverse=True)
                
            result.action_items = result.action_items[:self.config.max_items]
            
            # Calculate statistics
            result.items_found = len(result.action_items)
            result.assignees_identified = list(set(item.assignee for item in result.action_items))
            result.deadlines_found = sum(1 for item in result.action_items if item.deadline)
            result.priorities_assigned = sum(1 for item in result.action_items if item.priority != Priority.MEDIUM)
            
            # Finalize result
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = True
            
            logger.info(f"Extracted {result.items_found} action items")
            return result
            
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            
            result = ExtractorResult()
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = False
            result.error_message = str(e)
            
            return result
            
    def _find_action_patterns(self, transcript: str) -> List[tuple]:
        """Find potential action items using patterns"""
        potential_items = []
        
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, transcript)
            
            for match in matches:
                item_text = match.group(0)
                start_pos = match.start()
                
                # Extract context
                context_start = max(0, start_pos - self.config.context_window)
                context_end = min(len(transcript), start_pos + len(item_text) + self.config.context_window)
                context = transcript[context_start:context_end]
                
                potential_items.append((item_text, context))
                
        return potential_items
        
    def _process_action_item(self, item_text: str, context: str, 
                           meeting_id: Optional[str]) -> Optional[ActionItem]:
        """Process a potential action item"""
        try:
            # Create action item
            action_item = ActionItem(
                id=str(uuid.uuid4()),
                description=self._clean_description(item_text),
                assignee=self._extract_assignee(item_text, context),
                context=context,
                meeting_reference=meeting_id or "",
                source_text=item_text
            )
            
            # Extract additional information
            if self.config.detect_deadlines:
                action_item.deadline = self._extract_deadline(item_text, context)
                
            if self.config.detect_priorities:
                action_item.priority = self._extract_priority(item_text, context)
                
            if self.config.detect_dependencies:
                action_item.dependencies = self._extract_dependencies(item_text, context)
                
            # Calculate confidence
            action_item.confidence = self._calculate_confidence(action_item, item_text, context)
            
            return action_item
            
        except Exception as e:
            logger.error(f"Failed to process action item: {e}")
            return None
            
    def _clean_description(self, text: str) -> str:
        """Clean and normalize action item description"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common prefixes
        text = re.sub(r'^(will|going to|plan to|need to|should|must|have to)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(action|task|todo|follow up|next step):\s*', '', text, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        return text
        
    def _extract_assignee(self, item_text: str, context: str) -> str:
        """Extract assignee from text"""
        # Look for explicit assignment patterns
        assignment_patterns = [
            r"(?i)(\w+)\s+(will|should|needs to|has to)",
            r"(?i)(assigned to|responsibility of|owned by)\s+(\w+)",
            r"(?i)(\w+)\s+is\s+responsible",
            r"(?i)(\w+)\s+to\s+(do|handle|manage|work on)"
        ]
        
        for pattern in assignment_patterns:
            match = re.search(pattern, item_text + " " + context)
            if match:
                potential_assignee = match.group(1).lower()
                if potential_assignee in self.common_names:
                    return potential_assignee.capitalize()
                    
        # Look for speaker labels in context
        speaker_match = re.search(r"(\w+):\s*" + re.escape(item_text[:20]), context)
        if speaker_match:
            return speaker_match.group(1).capitalize()
            
        return "Unassigned"
        
    def _extract_deadline(self, item_text: str, context: str) -> Optional[datetime]:
        """Extract deadline from text"""
        text = item_text + " " + context
        
        for pattern in self.deadline_patterns:
            match = re.search(pattern, text)
            if match:
                return self._parse_deadline(match.group(0))
                
        return None
        
    def _parse_deadline(self, deadline_text: str) -> Optional[datetime]:
        """Parse deadline text into datetime"""
        try:
            # Simple deadline parsing (would need more sophisticated parsing in production)
            if "week" in deadline_text.lower():
                return datetime.now() + timedelta(weeks=1)
            elif "month" in deadline_text.lower():
                return datetime.now() + timedelta(days=30)
            elif "day" in deadline_text.lower():
                return datetime.now() + timedelta(days=1)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse deadline: {e}")
            return None
            
    def _extract_priority(self, item_text: str, context: str) -> Priority:
        """Extract priority from text"""
        text = (item_text + " " + context).lower()
        
        for priority, indicators in self.priority_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    return priority
                    
        return Priority.MEDIUM
        
    def _extract_dependencies(self, item_text: str, context: str) -> List[str]:
        """Extract dependencies from text"""
        dependencies = []
        
        # Look for dependency patterns
        text = item_text + " " + context
        dependency_patterns = [
            r"(?i)after\s+(\w+)",
            r"(?i)depends on\s+(\w+)",
            r"(?i)waiting for\s+(\w+)",
            r"(?i)blocked by\s+(\w+)"
        ]
        
        for pattern in dependency_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                dependencies.append(match.group(1))
                
        return dependencies
        
    def _calculate_confidence(self, action_item: ActionItem, 
                            item_text: str, context: str) -> float:
        """Calculate confidence score for action item"""
        confidence = 0.5  # Base confidence
        
        # Boost for explicit action words
        action_words = ["will", "should", "need", "must", "task", "action", "todo"]
        for word in action_words:
            if word in item_text.lower():
                confidence += 0.1
                
        # Boost for assignee
        if action_item.assignee != "Unassigned":
            confidence += 0.2
            
        # Boost for deadline
        if action_item.deadline:
            confidence += 0.1
            
        # Boost for priority indicators
        if action_item.priority != Priority.MEDIUM:
            confidence += 0.1
            
        # Penalty for very short descriptions
        if len(action_item.description) < 10:
            confidence -= 0.2
            
        return min(1.0, max(0.0, confidence))
        
    def _priority_score(self, priority: Priority) -> int:
        """Convert priority to numeric score for sorting"""
        return {
            Priority.URGENT: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1
        }.get(priority, 2)
        
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status"""
        return {
            "config": self.config.to_dict(),
            "patterns_loaded": len(self.action_patterns),
            "priority_indicators": len(self.priority_indicators),
            "deadline_patterns": len(self.deadline_patterns),
            "common_names": len(self.common_names)
        }


# Factory functions
def create_meeting_extractor() -> ActionItemExtractor:
    """Create extractor for general meetings"""
    config = ExtractorConfig(
        max_items=10,
        min_confidence=0.6,
        include_implied_actions=True,
        detect_deadlines=True,
        detect_priorities=True,
        sort_by_priority=True
    )
    return ActionItemExtractor(config)


def create_interview_extractor() -> ActionItemExtractor:
    """Create extractor for interview follow-ups"""
    config = ExtractorConfig(
        max_items=5,
        min_confidence=0.7,
        include_implied_actions=False,
        detect_deadlines=True,
        detect_priorities=False,
        sort_by_priority=False
    )
    return ActionItemExtractor(config)


def create_standup_extractor() -> ActionItemExtractor:
    """Create extractor for standup meetings"""
    config = ExtractorConfig(
        max_items=8,
        min_confidence=0.5,
        include_implied_actions=True,
        detect_deadlines=True,
        detect_priorities=True,
        sort_by_priority=True,
        group_by_assignee=True
    )
    return ActionItemExtractor(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Action Item Extractor Test")
    parser.add_argument("--transcript", type=str, required=True, help="Meeting transcript")
    parser.add_argument("--max-items", type=int, default=10, help="Maximum items to extract")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence")
    args = parser.parse_args()
    
    # Create extractor
    extractor = create_meeting_extractor()
    extractor.config.max_items = args.max_items
    extractor.config.min_confidence = args.min_confidence
    
    try:
        print(f"Extractor status: {extractor.get_status()}")
        
        # Extract action items
        print(f"Extracting action items from transcript...")
        result = extractor.extract_action_items(args.transcript)
        
        if result.success:
            print(f"Extraction completed successfully!")
            print(f"Items found: {result.items_found}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Assignees: {', '.join(result.assignees_identified)}")
            print(f"Deadlines found: {result.deadlines_found}")
            
            print(f"\nAction Items:")
            for i, item in enumerate(result.action_items, 1):
                print(f"{i}. {item.description}")
                print(f"   Assignee: {item.assignee}")
                print(f"   Priority: {item.priority.value}")
                print(f"   Confidence: {item.confidence:.3f}")
                if item.deadline:
                    print(f"   Deadline: {item.deadline.strftime('%Y-%m-%d')}")
                print()
                
        else:
            print(f"Extraction failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")