#!/usr/bin/env python3
"""
LLM Module - Local LLM Analysis System

Complete LLM processing module with local Phi-3 Mini integration for meeting analysis,
summarization, action item extraction, and topic identification.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Callable, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__description__ = "LLM processing module with local Phi-3 Mini integration"

# Import core components
try:
    from .local_llm_processor import (
        LocalLLMProcessor,
        LLMConfig,
        LLMResult,
        ModelType,
        ProcessingMode,
        OutputFormat,
        create_phi3_processor,
        create_optimized_processor,
        create_meeting_processor
    )
    
    from .prompt_templates import (
        PromptTemplateManager,
        PromptTemplate,
        TemplateConfig,
        TemplateType,
        MeetingType,
        create_meeting_templates,
        create_analysis_templates,
        create_standup_templates
    )
    
    from .meeting_analyzer import (
        MeetingAnalyzer,
        AnalysisConfig,
        AnalysisResult,
        AnalysisType,
        MeetingPhase,
        create_comprehensive_analyzer,
        create_quick_analyzer,
        create_standup_analyzer
    )
    
    from .action_item_extractor import (
        ActionItemExtractor,
        ActionItem,
        ExtractorConfig,
        ExtractorResult,
        Priority,
        Status,
        create_meeting_extractor,
        create_interview_extractor,
        create_standup_extractor
    )
    
    from .topic_identifier import (
        TopicIdentifier,
        Topic,
        TopicConfig,
        TopicResult,
        TopicType,
        ImportanceLevel,
        create_meeting_topic_identifier,
        create_discussion_topic_identifier,
        create_simple_topic_identifier
    )
    
    from .output_formatter import (
        OutputFormatter,
        FormattedOutput,
        OutputConfig,
        OutputFormat as FormatterOutputFormat,
        TemplateType as FormatterTemplateType,
        create_json_formatter,
        create_markdown_formatter,
        create_html_formatter
    )
    
    logger.info("All LLM components imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import LLM components: {e}")
    raise


# Main LLM system integration class
class LLMAnalysisSystem:
    """
    Complete LLM analysis system integrating all components
    """
    
    def __init__(self,
                 llm_config: Optional[LLMConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None,
                 extractor_config: Optional[ExtractorConfig] = None,
                 topic_config: Optional[TopicConfig] = None,
                 formatter_config: Optional[OutputConfig] = None):
        
        # Initialize components
        self.llm_processor = None
        self.meeting_analyzer = None
        self.action_extractor = None
        self.topic_identifier = None
        self.output_formatter = None
        self.template_manager = None
        
        # Configuration
        self.llm_config = llm_config or LLMConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        self.extractor_config = extractor_config or ExtractorConfig()
        self.topic_config = topic_config or TopicConfig()
        self.formatter_config = formatter_config or OutputConfig()
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "total_extractions": 0,
            "total_topic_identifications": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "error_count": 0,
            "uptime": 0.0
        }
        
        # Callbacks
        self.analysis_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        logger.info("LLMAnalysisSystem initialized")
        
    def initialize(self) -> bool:
        """Initialize the complete LLM analysis system"""
        try:
            logger.info("Initializing LLM analysis system...")
            
            # Initialize LLM processor
            self.llm_processor = LocalLLMProcessor(self.llm_config)
            if not self.llm_processor.initialize():
                logger.error("Failed to initialize LLM processor")
                return False
                
            # Initialize template manager
            self.template_manager = PromptTemplateManager()
            
            # Initialize meeting analyzer
            self.meeting_analyzer = MeetingAnalyzer(self.analysis_config)
            if not self.meeting_analyzer.initialize(self.llm_processor):
                logger.error("Failed to initialize meeting analyzer")
                return False
                
            # Initialize action item extractor
            self.action_extractor = ActionItemExtractor(self.extractor_config)
            
            # Initialize topic identifier
            self.topic_identifier = TopicIdentifier(self.topic_config)
            
            # Initialize output formatter
            self.output_formatter = OutputFormatter(self.formatter_config)
            
            # Set up component integration
            self._setup_component_integration()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("LLM analysis system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM system: {e}")
            return False
            
    def _setup_component_integration(self):
        """Set up integration between components"""
        # Add result callbacks
        if self.meeting_analyzer:
            self.meeting_analyzer.analysis_cache = {}  # Reset cache
            
        # Add error callbacks
        if self.llm_processor:
            self.llm_processor.add_error_callback(self._handle_llm_error)
            
    def _handle_llm_error(self, error_message: str):
        """Handle LLM processing errors"""
        logger.error(f"LLM processing error: {error_message}")
        self.stats["error_count"] += 1
        
        # Call user error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
    def analyze_meeting(self, transcript: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[AnalysisResult]:
        """Analyze meeting with all components"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return None
            
        try:
            # Perform comprehensive analysis
            result = self.meeting_analyzer.analyze_meeting(transcript, metadata)
            
            if result.success:
                # Update system statistics
                self.stats["total_analyses"] += 1
                self.stats["total_processing_time"] += result.processing_time
                
                # Update average confidence
                count = self.stats["total_analyses"]
                old_avg = self.stats["average_confidence"]
                self.stats["average_confidence"] = (
                    old_avg * (count - 1) + result.confidence
                ) / count
                
                # Call analysis callbacks
                for callback in self.analysis_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in analysis callback: {e}")
                        
            return result
            
        except Exception as e:
            logger.error(f"Meeting analysis failed: {e}")
            self.stats["error_count"] += 1
            return None
            
    def extract_action_items(self, transcript: str,
                           meeting_id: Optional[str] = None) -> Optional[ExtractorResult]:
        """Extract action items from transcript"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return None
            
        try:
            result = self.action_extractor.extract_action_items(transcript, meeting_id)
            
            if result.success:
                self.stats["total_extractions"] += 1
                
            return result
            
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            self.stats["error_count"] += 1
            return None
            
    def identify_topics(self, transcript: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[TopicResult]:
        """Identify topics from transcript"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return None
            
        try:
            result = self.topic_identifier.identify_topics(transcript, metadata)
            
            if result.success:
                self.stats["total_topic_identifications"] += 1
                
            return result
            
        except Exception as e:
            logger.error(f"Topic identification failed: {e}")
            self.stats["error_count"] += 1
            return None
            
    def generate_summary(self, transcript: str, 
                        template_id: str = "meeting_summary") -> Optional[LLMResult]:
        """Generate summary using LLM"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return None
            
        try:
            # Generate prompt
            variables = {"transcript": transcript}
            prompt = self.template_manager.generate_prompt(template_id, variables)
            
            # Generate summary
            result = self.llm_processor.generate(prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            self.stats["error_count"] += 1
            return None
            
    def format_analysis_results(self, analysis_result: AnalysisResult,
                              output_format: FormatterOutputFormat = FormatterOutputFormat.JSON) -> Optional[FormattedOutput]:
        """Format analysis results"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return None
            
        try:
            # Update formatter config
            self.output_formatter.config.output_format = output_format
            
            # Convert analysis result to dict
            data = analysis_result.to_dict()
            
            # Format output
            return self.output_formatter.format_output(data)
            
        except Exception as e:
            logger.error(f"Output formatting failed: {e}")
            self.stats["error_count"] += 1
            return None
            
    def complete_analysis_workflow(self, transcript: str,
                                 output_format: FormatterOutputFormat = FormatterOutputFormat.JSON,
                                 save_to_file: Optional[str] = None) -> Dict[str, Any]:
        """Complete analysis workflow with all components"""
        if not self.is_initialized:
            logger.error("LLM system not initialized")
            return {"success": False, "error": "System not initialized"}
            
        try:
            workflow_result = {
                "success": True,
                "analysis_result": None,
                "action_items": None,
                "topics": None,
                "formatted_output": None,
                "file_saved": False
            }
            
            # Perform meeting analysis
            analysis_result = self.analyze_meeting(transcript)
            if analysis_result:
                workflow_result["analysis_result"] = analysis_result.to_dict()
                
            # Extract action items
            action_result = self.extract_action_items(transcript)
            if action_result:
                workflow_result["action_items"] = action_result.to_dict()
                
            # Identify topics
            topic_result = self.identify_topics(transcript)
            if topic_result:
                workflow_result["topics"] = topic_result.to_dict()
                
            # Format output
            if analysis_result:
                formatted_output = self.format_analysis_results(analysis_result, output_format)
                if formatted_output:
                    workflow_result["formatted_output"] = formatted_output.to_dict()
                    
                    # Save to file if requested
                    if save_to_file:
                        if self.output_formatter.save_to_file(formatted_output, save_to_file):
                            workflow_result["file_saved"] = True
                            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Complete analysis workflow failed: {e}")
            self.stats["error_count"] += 1
            return {"success": False, "error": str(e)}
            
    def add_analysis_callback(self, callback: Callable):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable):
        """Add callback for error handling"""
        self.error_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "llm_system": self.stats,
            "llm_processor": self.llm_processor.get_stats() if self.llm_processor else {},
            "meeting_analyzer": self.meeting_analyzer.get_stats() if self.meeting_analyzer else {},
            "action_extractor": self.action_extractor.get_status() if self.action_extractor else {},
            "topic_identifier": self.topic_identifier.get_status() if self.topic_identifier else {},
            "output_formatter": self.output_formatter.get_status() if self.output_formatter else {}
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "components": {
                "llm_processor": self.llm_processor.get_model_info() if self.llm_processor else None,
                "meeting_analyzer": self.meeting_analyzer.get_status() if self.meeting_analyzer else None,
                "action_extractor": self.action_extractor.get_status() if self.action_extractor else None,
                "topic_identifier": self.topic_identifier.get_status() if self.topic_identifier else None,
                "output_formatter": self.output_formatter.get_status() if self.output_formatter else None
            },
            "stats": self.get_stats()
        }
        
    def shutdown(self):
        """Shutdown the LLM analysis system"""
        logger.info("Shutting down LLM analysis system...")
        
        self.is_running = False
        
        # Shutdown components
        if self.llm_processor:
            self.llm_processor.shutdown()
            
        if self.meeting_analyzer:
            self.meeting_analyzer.shutdown()
            
        logger.info("LLM analysis system shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions for common configurations
def create_meeting_llm_system(max_topics: int = 5,
                             max_action_items: int = 10,
                             output_format: FormatterOutputFormat = FormatterOutputFormat.JSON) -> LLMAnalysisSystem:
    """Create LLM system optimized for meetings"""
    
    # LLM config
    llm_config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        max_new_tokens=512,
        temperature=0.6,
        device="cpu"
    )
    
    # Analysis config
    analysis_config = AnalysisConfig(
        analysis_type=AnalysisType.SUMMARY,
        meeting_type=MeetingType.GENERAL,
        include_summary=True,
        include_action_items=True,
        include_topics=True,
        include_participants=True,
        max_action_items=max_action_items,
        max_topics=max_topics
    )
    
    # Extractor config
    extractor_config = ExtractorConfig(
        max_items=max_action_items,
        min_confidence=0.6,
        detect_deadlines=True,
        detect_priorities=True,
        sort_by_priority=True
    )
    
    # Topic config
    topic_config = TopicConfig(
        max_topics=max_topics,
        min_relevance_score=0.3,
        use_clustering=True,
        include_speaker_analysis=True
    )
    
    # Formatter config
    formatter_config = OutputConfig(
        output_format=output_format,
        template_type=FormatterTemplateType.FULL_REPORT,
        include_metadata=True,
        include_timestamps=True,
        pretty_print=True
    )
    
    return LLMAnalysisSystem(
        llm_config=llm_config,
        analysis_config=analysis_config,
        extractor_config=extractor_config,
        topic_config=topic_config,
        formatter_config=formatter_config
    )


def create_analysis_llm_system(output_format: FormatterOutputFormat = FormatterOutputFormat.JSON) -> LLMAnalysisSystem:
    """Create LLM system for detailed analysis"""
    
    # LLM config
    llm_config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        max_new_tokens=1024,
        temperature=0.5,
        device="cpu"
    )
    
    # Analysis config
    analysis_config = AnalysisConfig(
        analysis_type=AnalysisType.COMPREHENSIVE,
        include_summary=True,
        include_action_items=True,
        include_topics=True,
        include_participants=True,
        include_insights=True,
        max_summary_length=1000,
        max_action_items=15,
        max_topics=8
    )
    
    # Extractor config
    extractor_config = ExtractorConfig(
        max_items=15,
        min_confidence=0.7,
        include_implied_actions=True,
        detect_deadlines=True,
        detect_priorities=True,
        detect_dependencies=True
    )
    
    # Topic config
    topic_config = TopicConfig(
        max_topics=8,
        min_relevance_score=0.2,
        use_clustering=True,
        n_clusters=6,
        include_speaker_analysis=True
    )
    
    # Formatter config
    formatter_config = OutputConfig(
        output_format=output_format,
        template_type=FormatterTemplateType.FULL_REPORT,
        include_metadata=True,
        include_timestamps=True,
        include_confidence_scores=True,
        pretty_print=True
    )
    
    return LLMAnalysisSystem(
        llm_config=llm_config,
        analysis_config=analysis_config,
        extractor_config=extractor_config,
        topic_config=topic_config,
        formatter_config=formatter_config
    )


def create_standup_llm_system(output_format: FormatterOutputFormat = FormatterOutputFormat.TEXT) -> LLMAnalysisSystem:
    """Create LLM system optimized for standup meetings"""
    
    # LLM config
    llm_config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        max_new_tokens=256,
        temperature=0.5,
        device="cpu"
    )
    
    # Analysis config
    analysis_config = AnalysisConfig(
        analysis_type=AnalysisType.QUICK,
        meeting_type=MeetingType.STANDUP,
        include_summary=True,
        include_action_items=True,
        include_topics=False,
        include_participants=True,
        max_summary_length=200,
        max_action_items=8
    )
    
    # Extractor config
    extractor_config = ExtractorConfig(
        max_items=8,
        min_confidence=0.5,
        detect_deadlines=True,
        detect_priorities=True,
        group_by_assignee=True
    )
    
    # Topic config
    topic_config = TopicConfig(
        max_topics=3,
        min_relevance_score=0.4,
        use_clustering=False,
        include_speaker_analysis=True
    )
    
    # Formatter config
    formatter_config = OutputConfig(
        output_format=output_format,
        template_type=FormatterTemplateType.SUMMARY,
        include_metadata=False,
        include_timestamps=False,
        pretty_print=True
    )
    
    return LLMAnalysisSystem(
        llm_config=llm_config,
        analysis_config=analysis_config,
        extractor_config=extractor_config,
        topic_config=topic_config,
        formatter_config=formatter_config
    )


# Export all public components
__all__ = [
    # Main system
    "LLMAnalysisSystem",
    
    # Core components
    "LocalLLMProcessor",
    "PromptTemplateManager",
    "MeetingAnalyzer",
    "ActionItemExtractor",
    "TopicIdentifier",
    "OutputFormatter",
    
    # Configuration classes
    "LLMConfig",
    "TemplateConfig",
    "AnalysisConfig",
    "ExtractorConfig",
    "TopicConfig",
    "OutputConfig",
    
    # Result classes
    "LLMResult",
    "AnalysisResult",
    "ExtractorResult",
    "TopicResult",
    "FormattedOutput",
    "ActionItem",
    "Topic",
    "PromptTemplate",
    
    # Enums
    "ModelType",
    "ProcessingMode",
    "TemplateType",
    "MeetingType",
    "AnalysisType",
    "Priority",
    "Status",
    "TopicType",
    "ImportanceLevel",
    "OutputFormat",
    "FormatterOutputFormat",
    "FormatterTemplateType",
    
    # Factory functions
    "create_meeting_llm_system",
    "create_analysis_llm_system",
    "create_standup_llm_system",
    "create_phi3_processor",
    "create_optimized_processor",
    "create_meeting_processor",
    "create_meeting_templates",
    "create_analysis_templates",
    "create_comprehensive_analyzer",
    "create_quick_analyzer",
    "create_meeting_extractor",
    "create_interview_extractor",
    "create_meeting_topic_identifier",
    "create_discussion_topic_identifier",
    "create_json_formatter",
    "create_markdown_formatter",
    "create_html_formatter",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]


# Initialize module
logger.info(f"LLM module {__version__} loaded successfully")
logger.info(f"Available components: {len(__all__)} exports")