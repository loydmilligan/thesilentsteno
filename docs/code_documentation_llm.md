# LLM Module Documentation

## Module Overview

The LLM module provides comprehensive local language model processing capabilities for The Silent Steno, implementing Microsoft's Phi-3 Mini model optimized for Raspberry Pi 5 deployment. The module offers meeting analysis, action item extraction, topic identification, and multi-format output generation with advanced prompt engineering and caching systems.

## Dependencies

### External Dependencies
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model inference
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities (optional)
- `threading` - Thread management
- `re` - Regular expressions
- `json` - JSON processing
- `yaml` - YAML processing
- `xml.etree.ElementTree` - XML processing
- `datetime` - Date/time operations
- `pathlib` - Path operations
- `logging` - Logging system
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `hashlib` - Hash functions
- `time` - Timing operations

### Internal Dependencies
- `src.core.events` - Event system
- `src.core.logging` - Logging system
- `src.core.monitoring` - Performance monitoring
- `src.core.config` - Configuration management

## Architecture Overview

### Processing Pipeline
```
Transcript → LLM Processor → Analysis Engine → Output Formatter → Multi-Format Results
                ↓
         Prompt Templates → Context Management → Quality Assessment
```

### Model Configuration
- **Primary Model**: Microsoft Phi-3 Mini (4K context)
- **Hardware Target**: Raspberry Pi 5 (4 cores, CPU-only)
- **Memory Optimization**: float32 precision, batch size 1
- **Context Management**: 3000 token input limit with smart truncation

## File Documentation

### 1. `__init__.py`

**Purpose**: Main integration layer orchestrating all LLM components with caching and workflow management.

#### Classes

##### `LLMAnalysisSystem`
Central LLM analysis system coordinating all components.

**Attributes:**
- `llm_processor: LocalLLMProcessor` - Core LLM processing engine
- `meeting_analyzer: MeetingAnalyzer` - Meeting analysis component
- `action_extractor: ActionItemExtractor` - Action item extraction
- `topic_identifier: TopicIdentifier` - Topic identification
- `output_formatter: OutputFormatter` - Output formatting
- `cache_enabled: bool` - Enable result caching
- `statistics: dict` - Performance statistics

**Methods:**
- `__init__(config: dict = None)` - Initialize LLM analysis system
- `analyze_meeting(transcript: str, meeting_type: str = "general")` - Analyze complete meeting
- `analyze_component(transcript: str, component: str, options: dict = None)` - Analyze specific component
- `get_analysis_cache(cache_key: str)` - Get cached analysis result
- `clear_cache()` - Clear analysis cache
- `get_statistics()` - Get performance statistics
- `set_callback(event_type: str, callback: callable)` - Set event callback

#### Factory Functions

##### `create_meeting_analysis_system(config: dict = None) -> LLMAnalysisSystem`
Create system optimized for general meeting analysis.

##### `create_standup_analysis_system(config: dict = None) -> LLMAnalysisSystem`
Create system optimized for standup meeting analysis.

##### `create_planning_analysis_system(config: dict = None) -> LLMAnalysisSystem`
Create system optimized for planning meeting analysis.

**Usage Example:**
```python
from src.llm import create_meeting_analysis_system

# Create meeting analysis system
analysis_system = create_meeting_analysis_system({
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "max_tokens": 3000,
    "temperature": 0.7,
    "enable_caching": True
})

# Analyze complete meeting
transcript = "Meeting transcript content..."
results = analysis_system.analyze_meeting(transcript, meeting_type="general")

print(f"Summary: {results['summary']}")
print(f"Action Items: {len(results['action_items'])}")
print(f"Topics: {results['topics']}")
print(f"Participants: {results['participants']}")

# Analyze specific component
action_items = analysis_system.analyze_component(
    transcript, 
    "action_items", 
    options={"include_assignments": True, "include_deadlines": True}
)

# Get statistics
stats = analysis_system.get_statistics()
print(f"Total analyses: {stats['total_analyses']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Average processing time: {stats['average_processing_time']:.2f}s")
```

### 2. `local_llm_processor.py`

**Purpose**: Core LLM processing engine with Phi-3 Mini model integration and Raspberry Pi 5 optimization.

#### Classes

##### `LLMConfig`
LLM processor configuration.

**Attributes:**
- `model_name: str` - Model identifier
- `max_tokens: int` - Maximum generation tokens
- `temperature: float` - Generation temperature
- `top_p: float` - Top-p sampling
- `top_k: int` - Top-k sampling
- `device: str` - Compute device ("cpu" for Pi 5)
- `torch_dtype: str` - PyTorch data type
- `num_threads: int` - Number of threads
- `batch_size: int` - Batch size
- `context_length: int` - Context length limit

##### `GenerationResult`
LLM generation result.

**Attributes:**
- `generated_text: str` - Generated text
- `input_tokens: int` - Input token count
- `output_tokens: int` - Output token count
- `generation_time: float` - Generation time in seconds
- `quality_score: float` - Quality assessment score
- `truncated: bool` - Whether input was truncated

##### `LocalLLMProcessor`
Main local LLM processing engine.

**Methods:**
- `__init__(config: LLMConfig)` - Initialize LLM processor
- `load_model()` - Load Phi-3 model
- `unload_model()` - Unload model from memory
- `generate_text(prompt: str, max_tokens: int = None)` - Generate text
- `generate_with_chat_template(messages: List[dict])` - Generate with chat format
- `analyze_text_quality(text: str)` - Analyze generated text quality
- `get_model_info()` - Get model information
- `optimize_for_pi5()` - Optimize for Raspberry Pi 5
- `get_performance_metrics()` - Get performance metrics

**Usage Example:**
```python
from src.llm.local_llm_processor import LocalLLMProcessor, LLMConfig

# Create LLM configuration for Pi 5
config = LLMConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_tokens=512,
    temperature=0.7,
    device="cpu",
    torch_dtype="float32",
    num_threads=4,
    batch_size=1
)

# Create processor
llm_processor = LocalLLMProcessor(config)

# Load model
llm_processor.load_model()

# Optimize for Pi 5
llm_processor.optimize_for_pi5()

# Generate text
prompt = "Analyze this meeting transcript and provide a summary:"
result = llm_processor.generate_text(prompt + transcript)

print(f"Generated text: {result.generated_text}")
print(f"Generation time: {result.generation_time:.2f}s")
print(f"Quality score: {result.quality_score:.2f}")

# Use chat template
messages = [
    {"role": "system", "content": "You are a meeting analysis assistant."},
    {"role": "user", "content": f"Analyze this transcript: {transcript}"}
]

chat_result = llm_processor.generate_with_chat_template(messages)
print(f"Chat response: {chat_result.generated_text}")

# Get performance metrics
metrics = llm_processor.get_performance_metrics()
print(f"Average generation time: {metrics['avg_generation_time']:.2f}s")
print(f"Tokens per second: {metrics['tokens_per_second']:.1f}")
```

### 3. `meeting_analyzer.py`

**Purpose**: Comprehensive meeting transcript analysis with multi-component processing and quality assessment.

#### Classes

##### `MeetingAnalysisConfig`
Meeting analysis configuration.

**Attributes:**
- `analysis_depth: str` - Analysis depth ("quick", "standard", "comprehensive")
- `meeting_type: str` - Meeting type ("general", "standup", "planning", "retrospective")
- `enable_summary: bool` - Enable summary generation
- `enable_action_items: bool` - Enable action item extraction
- `enable_topics: bool` - Enable topic identification
- `enable_participants: bool` - Enable participant analysis
- `enable_insights: bool` - Enable insight extraction
- `chunk_size: int` - Transcript chunk size
- `quality_threshold: float` - Quality threshold

##### `MeetingAnalysisResult`
Comprehensive meeting analysis result.

**Attributes:**
- `summary: str` - Meeting summary
- `action_items: List[dict]` - Extracted action items
- `topics: List[str]` - Identified topics
- `participants: List[dict]` - Participant analysis
- `insights: List[str]` - Key insights
- `quality_metrics: dict` - Quality assessment
- `processing_time: float` - Analysis processing time
- `confidence_score: float` - Overall confidence

##### `MeetingAnalyzer`
Main meeting analysis engine.

**Methods:**
- `__init__(llm_processor: LocalLLMProcessor, config: MeetingAnalysisConfig)` - Initialize analyzer
- `analyze_meeting(transcript: str, meeting_type: str = None)` - Analyze complete meeting
- `analyze_summary(transcript: str)` - Generate meeting summary
- `analyze_participants(transcript: str)` - Analyze participants
- `analyze_insights(transcript: str)` - Extract key insights
- `chunk_transcript(transcript: str, chunk_size: int)` - Chunk large transcripts
- `merge_chunk_results(chunk_results: List[dict])` - Merge chunked analysis
- `assess_quality(result: MeetingAnalysisResult)` - Assess result quality

**Usage Example:**
```python
from src.llm.meeting_analyzer import MeetingAnalyzer, MeetingAnalysisConfig

# Create analysis configuration
config = MeetingAnalysisConfig(
    analysis_depth="comprehensive",
    meeting_type="general",
    enable_summary=True,
    enable_action_items=True,
    enable_topics=True,
    enable_participants=True,
    enable_insights=True,
    chunk_size=2000,
    quality_threshold=0.7
)

# Create analyzer
analyzer = MeetingAnalyzer(llm_processor, config)

# Analyze meeting
result = analyzer.analyze_meeting(transcript, meeting_type="general")

print(f"Summary: {result.summary}")
print(f"Action Items: {len(result.action_items)}")
for item in result.action_items:
    print(f"  - {item['description']} (assigned to: {item['assignee']})")

print(f"Topics: {result.topics}")
print(f"Participants: {len(result.participants)}")
for participant in result.participants:
    print(f"  - {participant['name']}: {participant['contribution']}")

print(f"Insights: {result.insights}")
print(f"Quality Score: {result.confidence_score:.2f}")
print(f"Processing Time: {result.processing_time:.2f}s")
```

### 4. `action_item_extractor.py`

**Purpose**: Intelligent extraction of action items from meeting transcripts with assignment and deadline detection.

#### Classes

##### `ActionItemConfig`
Action item extraction configuration.

**Attributes:**
- `include_assignments: bool` - Include assignee information
- `include_deadlines: bool` - Include deadline information
- `include_dependencies: bool` - Include task dependencies
- `confidence_threshold: float` - Minimum confidence threshold
- `priority_scoring: bool` - Enable priority scoring
- `context_window: int` - Context window size

##### `ActionItem`
Action item data structure.

**Attributes:**
- `description: str` - Action item description
- `assignee: str` - Assigned person
- `deadline: str` - Deadline information
- `priority: str` - Priority level ("low", "medium", "high")
- `status: str` - Status ("pending", "in_progress", "completed")
- `dependencies: List[str]` - Task dependencies
- `context: str` - Surrounding context
- `confidence: float` - Extraction confidence

##### `ActionItemExtractor`
Main action item extraction system.

**Methods:**
- `__init__(llm_processor: LocalLLMProcessor, config: ActionItemConfig)` - Initialize extractor
- `extract_action_items(transcript: str)` - Extract action items
- `identify_assignments(text: str)` - Identify assignees
- `detect_deadlines(text: str)` - Detect deadlines
- `calculate_priority(action_item: ActionItem)` - Calculate priority
- `find_dependencies(action_items: List[ActionItem])` - Find dependencies
- `validate_action_items(action_items: List[ActionItem])` - Validate results

**Usage Example:**
```python
from src.llm.action_item_extractor import ActionItemExtractor, ActionItemConfig

# Create extraction configuration
config = ActionItemConfig(
    include_assignments=True,
    include_deadlines=True,
    include_dependencies=True,
    confidence_threshold=0.6,
    priority_scoring=True,
    context_window=100
)

# Create extractor
extractor = ActionItemExtractor(llm_processor, config)

# Extract action items
action_items = extractor.extract_action_items(transcript)

print(f"Found {len(action_items)} action items:")
for item in action_items:
    print(f"  - {item.description}")
    print(f"    Assignee: {item.assignee}")
    print(f"    Deadline: {item.deadline}")
    print(f"    Priority: {item.priority}")
    print(f"    Confidence: {item.confidence:.2f}")
    if item.dependencies:
        print(f"    Dependencies: {', '.join(item.dependencies)}")
```

### 5. `topic_identifier.py`

**Purpose**: Identification and analysis of meeting topics with clustering and importance scoring.

#### Classes

##### `TopicConfig`
Topic identification configuration.

**Attributes:**
- `max_topics: int` - Maximum number of topics
- `min_topic_frequency: int` - Minimum topic frequency
- `enable_clustering: bool` - Enable topic clustering
- `clustering_method: str` - Clustering method
- `importance_threshold: float` - Importance threshold
- `context_window: int` - Context window size

##### `Topic`
Topic information structure.

**Attributes:**
- `name: str` - Topic name
- `keywords: List[str]` - Topic keywords
- `importance: float` - Importance score
- `frequency: int` - Topic frequency
- `context: List[str]` - Context snippets
- `participants: List[str]` - Discussing participants
- `timeline: List[tuple]` - Topic timeline

##### `TopicIdentifier`
Main topic identification system.

**Methods:**
- `__init__(llm_processor: LocalLLMProcessor, config: TopicConfig)` - Initialize identifier
- `identify_topics(transcript: str)` - Identify topics
- `extract_keywords(text: str)` - Extract keywords
- `cluster_topics(topics: List[Topic])` - Cluster related topics
- `calculate_importance(topic: Topic)` - Calculate importance
- `create_topic_timeline(transcript: str, topics: List[Topic])` - Create timeline
- `analyze_topic_relationships(topics: List[Topic])` - Analyze relationships

**Usage Example:**
```python
from src.llm.topic_identifier import TopicIdentifier, TopicConfig

# Create topic configuration
config = TopicConfig(
    max_topics=10,
    min_topic_frequency=2,
    enable_clustering=True,
    clustering_method="kmeans",
    importance_threshold=0.5,
    context_window=200
)

# Create identifier
identifier = TopicIdentifier(llm_processor, config)

# Identify topics
topics = identifier.identify_topics(transcript)

print(f"Identified {len(topics)} topics:")
for topic in topics:
    print(f"  - {topic.name} (importance: {topic.importance:.2f})")
    print(f"    Keywords: {', '.join(topic.keywords)}")
    print(f"    Participants: {', '.join(topic.participants)}")
    print(f"    Frequency: {topic.frequency}")
```

### 6. `prompt_templates.py`

**Purpose**: Comprehensive prompt template system for different analysis tasks with variable substitution.

#### Classes

##### `PromptTemplate`
Prompt template definition.

**Attributes:**
- `name: str` - Template name
- `template: str` - Template string
- `variables: List[str]` - Required variables
- `description: str` - Template description
- `category: str` - Template category
- `meeting_type: str` - Target meeting type

##### `PromptTemplateManager`
Main template management system.

**Methods:**
- `__init__(config: dict = None)` - Initialize template manager
- `get_template(name: str)` - Get template by name
- `register_template(template: PromptTemplate)` - Register new template
- `render_template(name: str, variables: dict)` - Render template with variables
- `list_templates(category: str = None)` - List available templates
- `recommend_template(meeting_type: str, task: str)` - Recommend template
- `validate_template(template: PromptTemplate)` - Validate template

#### Built-in Templates

##### Meeting Analysis Templates
- `meeting_summary` - General meeting summary
- `standup_summary` - Standup meeting summary
- `planning_summary` - Planning meeting summary
- `retrospective_summary` - Retrospective meeting summary

##### Action Item Templates
- `action_item_extraction` - General action item extraction
- `task_assignment` - Task assignment identification
- `deadline_detection` - Deadline detection

##### Topic Analysis Templates
- `topic_identification` - Topic identification
- `theme_analysis` - Theme and pattern analysis
- `discussion_analysis` - Discussion point analysis

**Usage Example:**
```python
from src.llm.prompt_templates import PromptTemplateManager

# Create template manager
template_manager = PromptTemplateManager()

# Get template
template = template_manager.get_template("meeting_summary")

# Render template with variables
variables = {
    "transcript": transcript,
    "meeting_type": "general",
    "participants": ["Alice", "Bob", "Charlie"]
}

rendered_prompt = template_manager.render_template("meeting_summary", variables)

# Use rendered prompt with LLM
result = llm_processor.generate_text(rendered_prompt)

# Get template recommendation
recommended = template_manager.recommend_template("standup", "summary")
print(f"Recommended template: {recommended.name}")
```

### 7. `output_formatter.py`

**Purpose**: Multi-format output generation with template-based formatting and validation.

#### Classes

##### `FormatterConfig`
Output formatter configuration.

**Attributes:**
- `default_format: str` - Default output format
- `include_metadata: bool` - Include metadata
- `include_timestamps: bool` - Include timestamps
- `template_style: str` - Template style
- `validation_enabled: bool` - Enable validation
- `custom_templates: dict` - Custom templates

##### `FormattedOutput`
Formatted output result.

**Attributes:**
- `content: str` - Formatted content
- `format: str` - Output format
- `metadata: dict` - Output metadata
- `file_extension: str` - File extension
- `mime_type: str` - MIME type
- `size: int` - Content size

##### `OutputFormatter`
Main output formatting system.

**Methods:**
- `__init__(config: FormatterConfig)` - Initialize formatter
- `format_output(data: dict, format: str)` - Format output
- `format_json(data: dict)` - Format as JSON
- `format_markdown(data: dict)` - Format as Markdown
- `format_html(data: dict)` - Format as HTML
- `format_text(data: dict)` - Format as plain text
- `format_csv(data: dict)` - Format as CSV
- `format_xml(data: dict)` - Format as XML
- `format_yaml(data: dict)` - Format as YAML
- `save_output(formatted_output: FormattedOutput, file_path: str)` - Save to file

**Usage Example:**
```python
from src.llm.output_formatter import OutputFormatter, FormatterConfig

# Create formatter configuration
config = FormatterConfig(
    default_format="markdown",
    include_metadata=True,
    include_timestamps=True,
    template_style="professional",
    validation_enabled=True
)

# Create formatter
formatter = OutputFormatter(config)

# Format analysis results
analysis_data = {
    "summary": "Meeting summary...",
    "action_items": [{"description": "Task 1", "assignee": "Alice"}],
    "topics": ["Project timeline", "Budget review"],
    "participants": ["Alice", "Bob", "Charlie"]
}

# Format as different formats
markdown_output = formatter.format_markdown(analysis_data)
json_output = formatter.format_json(analysis_data)
html_output = formatter.format_html(analysis_data)

# Save outputs
formatter.save_output(markdown_output, "meeting_analysis.md")
formatter.save_output(json_output, "meeting_analysis.json")
formatter.save_output(html_output, "meeting_analysis.html")

print(f"Markdown output ({markdown_output.size} bytes):")
print(markdown_output.content[:500] + "...")
```

## Module Integration

The LLM module integrates with other Silent Steno components:

1. **AI Module**: Processes transcription results for analysis
2. **Data Module**: Stores analysis results and caching
3. **Export Module**: Provides formatted output for export
4. **Core Events**: Publishes analysis events and progress
5. **UI Module**: Displays analysis results in real-time

## Common Usage Patterns

### Complete Meeting Analysis Workflow
```python
# Initialize complete LLM analysis system
from src.llm import create_meeting_analysis_system

# Create system with comprehensive configuration
analysis_system = create_meeting_analysis_system({
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "max_tokens": 3000,
    "temperature": 0.7,
    "analysis_depth": "comprehensive",
    "enable_caching": True,
    "output_formats": ["json", "markdown", "html"]
})

# Process meeting transcript
transcript = load_meeting_transcript("meeting_2024_01_15.txt")

# Perform comprehensive analysis
results = analysis_system.analyze_meeting(transcript, meeting_type="general")

# Access analysis components
print("=== Meeting Analysis Results ===")
print(f"Summary: {results['summary']}")

print(f"\nAction Items ({len(results['action_items'])}):")
for i, item in enumerate(results['action_items'], 1):
    print(f"{i}. {item['description']}")
    print(f"   Assignee: {item['assignee']}")
    print(f"   Deadline: {item['deadline']}")
    print(f"   Priority: {item['priority']}")

print(f"\nKey Topics: {', '.join(results['topics'])}")

print(f"\nParticipants:")
for participant in results['participants']:
    print(f"- {participant['name']}: {participant['contribution']}")

print(f"\nInsights:")
for insight in results['insights']:
    print(f"- {insight}")

# Export results in multiple formats
analysis_system.export_results("json", "meeting_analysis.json")
analysis_system.export_results("markdown", "meeting_analysis.md")
analysis_system.export_results("html", "meeting_analysis.html")
```

### Component-Specific Analysis
```python
# Use individual components for specific analysis
from src.llm.action_item_extractor import ActionItemExtractor, ActionItemConfig
from src.llm.topic_identifier import TopicIdentifier, TopicConfig
from src.llm.local_llm_processor import LocalLLMProcessor, LLMConfig

# Initialize LLM processor
llm_config = LLMConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_tokens=512,
    temperature=0.7,
    device="cpu"
)
llm_processor = LocalLLMProcessor(llm_config)
llm_processor.load_model()

# Extract action items
action_config = ActionItemConfig(
    include_assignments=True,
    include_deadlines=True,
    priority_scoring=True
)
action_extractor = ActionItemExtractor(llm_processor, action_config)
action_items = action_extractor.extract_action_items(transcript)

# Identify topics
topic_config = TopicConfig(
    max_topics=10,
    enable_clustering=True,
    importance_threshold=0.5
)
topic_identifier = TopicIdentifier(llm_processor, topic_config)
topics = topic_identifier.identify_topics(transcript)

# Process results
print("Action Items:")
for item in action_items:
    print(f"- {item.description} (assigned to: {item.assignee})")

print("\nTopics:")
for topic in topics:
    print(f"- {topic.name} (importance: {topic.importance:.2f})")
```

### Performance Monitoring and Optimization
```python
# Monitor LLM performance and optimize for Pi 5
class LLMPerformanceMonitor:
    def __init__(self, llm_processor):
        self.llm_processor = llm_processor
        self.performance_log = []
        
    def monitor_generation(self, prompt, max_tokens=None):
        start_time = time.time()
        result = self.llm_processor.generate_text(prompt, max_tokens)
        end_time = time.time()
        
        # Log performance metrics
        metrics = {
            "generation_time": end_time - start_time,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.output_tokens / (end_time - start_time),
            "quality_score": result.quality_score,
            "timestamp": end_time
        }
        
        self.performance_log.append(metrics)
        return result
    
    def get_performance_summary(self):
        if not self.performance_log:
            return {}
        
        return {
            "total_generations": len(self.performance_log),
            "average_time": sum(m["generation_time"] for m in self.performance_log) / len(self.performance_log),
            "average_tokens_per_second": sum(m["tokens_per_second"] for m in self.performance_log) / len(self.performance_log),
            "average_quality": sum(m["quality_score"] for m in self.performance_log) / len(self.performance_log)
        }
    
    def optimize_performance(self):
        # Analyze performance patterns
        summary = self.get_performance_summary()
        
        # Adjust configuration based on performance
        if summary["average_time"] > 30:  # > 30 seconds
            print("Performance slow, reducing max_tokens")
            self.llm_processor.config.max_tokens = max(256, self.llm_processor.config.max_tokens - 100)
        
        if summary["average_quality"] < 0.7:
            print("Quality low, increasing temperature")
            self.llm_processor.config.temperature = min(1.0, self.llm_processor.config.temperature + 0.1)

# Use performance monitoring
monitor = LLMPerformanceMonitor(llm_processor)

# Monitor generation
result = monitor.monitor_generation("Analyze this meeting transcript...")

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Performance Summary: {summary}")

# Optimize based on performance
monitor.optimize_performance()
```

### Custom Template Development
```python
# Create custom prompt templates for specific use cases
from src.llm.prompt_templates import PromptTemplate, PromptTemplateManager

# Create custom template
custom_template = PromptTemplate(
    name="team_retrospective",
    template="""
    You are analyzing a team retrospective meeting transcript. 
    
    Meeting Type: {meeting_type}
    Duration: {duration} minutes
    Participants: {participants}
    
    Transcript:
    {transcript}
    
    Please analyze the retrospective and provide:
    1. What went well
    2. What could be improved
    3. Action items for next sprint
    4. Team sentiment analysis
    
    Format your response as structured output with clear sections.
    """,
    variables=["meeting_type", "duration", "participants", "transcript"],
    description="Template for team retrospective analysis",
    category="retrospective",
    meeting_type="retrospective"
)

# Register custom template
template_manager = PromptTemplateManager()
template_manager.register_template(custom_template)

# Use custom template
variables = {
    "meeting_type": "Sprint Retrospective",
    "duration": 45,
    "participants": ["Alice", "Bob", "Charlie", "Diana"],
    "transcript": retrospective_transcript
}

rendered_prompt = template_manager.render_template("team_retrospective", variables)
analysis_result = llm_processor.generate_text(rendered_prompt)

print(f"Retrospective Analysis: {analysis_result.generated_text}")
```

### Caching and Performance Optimization
```python
# Implement intelligent caching for repeated analysis
import hashlib
import json
import os

class LLMAnalysisCache:
    def __init__(self, cache_dir="cache/llm_analysis"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, transcript, analysis_type, config):
        # Create deterministic cache key
        cache_data = {
            "transcript_hash": hashlib.md5(transcript.encode()).hexdigest(),
            "analysis_type": analysis_type,
            "config": config
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def get_cached_result(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def cache_result(self, cache_key, result):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def clear_cache(self):
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, file))

# Use caching system
cache = LLMAnalysisCache()

def cached_analysis(transcript, analysis_type, analysis_config):
    # Check cache first
    cache_key = cache.get_cache_key(transcript, analysis_type, analysis_config)
    cached_result = cache.get_cached_result(cache_key)
    
    if cached_result:
        print("Using cached result")
        return cached_result
    
    # Perform analysis
    print("Performing new analysis")
    result = analysis_system.analyze_component(transcript, analysis_type, analysis_config)
    
    # Cache result
    cache.cache_result(cache_key, result)
    
    return result

# Use cached analysis
result = cached_analysis(transcript, "action_items", {"include_assignments": True})
```

This comprehensive LLM module provides sophisticated local language model capabilities optimized for Raspberry Pi 5 deployment, enabling advanced meeting analysis while maintaining data privacy and reducing latency through edge processing.