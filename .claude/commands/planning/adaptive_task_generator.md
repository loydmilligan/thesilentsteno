# Adaptive Task Generator with Complexity Scoring Command

```bash
claude-code "Generate development tasks with adaptive complexity breakdown based on scoring threshold.

## Task: Create Adaptive Development Task Breakdown

**Arguments:** [complexity_threshold] [detail_level] [estimation_method]

- **complexity_threshold**: Score 1-10 (tasks above this get broken into subtasks)
- **detail_level**: [basic/standard/comprehensive] - Overall detail level
- **estimation_method**: [story_points/hours/t_shirt] - Estimation approach

**Example Usage:**
- `claude-code adaptive_task_generator \"6 standard story_points\"` 
- `claude-code adaptive_task_generator \"4 comprehensive hours\"`
- `claude-code adaptive_task_generator \"7 basic t_shirt\"`

**Input Sources:** [product_requirements_document.json, ui_specification.json]

Start your response with: \"⚡ **ADAPTIVE_TASK_GENERATOR EXECUTING** - Creating complexity-scored task breakdown (threshold: [X], detail: [Y], estimation: [Z])\"

## Adaptive Task Generation Process:

### 1. Load Context and Parse Arguments
- Read PRD with user stories and acceptance criteria
- Load UI specifications for implementation context
- Parse complexity threshold, detail level, and estimation method
- Understand feature relationships and dependencies

### 2. Initial Task Analysis and Complexity Scoring

For each user story/epic, create main task and score complexity:

#### Complexity Scoring Criteria (1-10 Scale):

**Low Complexity (1-3): Simple Implementation**
- Basic CRUD operations
- Simple UI components (buttons, inputs, basic forms)
- Configuration files and setup
- Static content pages
- Basic routing

**Medium Complexity (4-6): Moderate Implementation**
- User authentication workflows
- Database schema with relationships
- API endpoints with validation
- Complex forms with validation
- Search and filtering functionality
- File upload/download

**High Complexity (7-10): Complex Implementation**
- AI integration and machine learning features
- Complex multi-step workflows
- Real-time features (websockets, live updates)
- Advanced algorithms and calculations
- Third-party API integrations
- Complex state management
- Performance optimization features

### 3. Adaptive Task Breakdown Strategy

#### For Tasks ≤ Complexity Threshold:
- **Single Task Approach**: Create comprehensive single task
- **Rich Context**: Include all implementation details in one task
- **Complete Specifications**: Full acceptance criteria, technical details, testing requirements
- **Implementation Ready**: Developer can execute without further breakdown

#### For Tasks > Complexity Threshold:
- **Subtask Decomposition**: Break into multiple smaller, focused tasks
- **Dependency Mapping**: Clear prerequisite relationships between subtasks
- **Incremental Implementation**: Each subtask builds toward complete feature
- **Integration Tasks**: Explicit tasks for combining subtask outputs

### 4. Comprehensive Task Generation

Create adaptive task structure based on complexity assessment:

```json
{
  \"task_generation_metadata\": {
    \"generated_date\": \"[ISO timestamp]\",
    \"complexity_threshold\": \"[Threshold used]\",
    \"detail_level\": \"[basic/standard/comprehensive]\",
    \"estimation_method\": \"[story_points/hours/t_shirt]\",
    \"source_documents\": [\"product_requirements_document.json\", \"ui_specification.json\"],
    \"total_main_tasks\": \"[Number of main tasks]\",
    \"tasks_requiring_breakdown\": \"[Number above threshold]\",
    \"total_implementation_tasks\": \"[Final task count after breakdown]\"
  },
  \"complexity_analysis\": [
    {
      \"user_story_id\": \"US-001\",
      \"story_title\": \"[User story title]\",
      \"initial_complexity_score\": \"[1-10]\",
      \"complexity_factors\": [
        \"[Factor 1: AI integration complexity]\",
        \"[Factor 2: Multi-step workflow]\",
        \"[Factor 3: External API dependencies]\"
      ],
      \"complexity_justification\": \"[Why this score was assigned]\",
      \"breakdown_decision\": \"[single_task/break_into_subtasks]\",
      \"breakdown_rationale\": \"[Why this decision was made]\"
    }
  ],
  \"implementation_tasks\": [
    {
      \"task_id\": \"TASK-001\",
      \"task_type\": \"[single_implementation/parent_task/subtask]\",
      \"parent_task_id\": \"[null if main task, parent ID if subtask]\",
      \"user_story_id\": \"US-001\",
      \"task_title\": \"[Clear, actionable task title]\",
      \"complexity_score\": \"[1-10]\",
      \"breakdown_reason\": \"[why_single_task/complexity_above_threshold]\",
      \"priority\": \"[Critical/High/Medium/Low]\",
      \"effort_estimate\": {
        \"[estimation_method]\": \"[Estimate value]\",
        \"confidence_level\": \"[High/Medium/Low]\",
        \"estimation_notes\": \"[Factors affecting estimate]\"
      },
      \"task_description\": {
        \"summary\": \"[Concise task description]\",
        \"detailed_scope\": \"[Comprehensive scope definition]\",
        \"business_value\": \"[Why this task delivers value]\",
        \"user_impact\": \"[How this affects end users]\"
      },
      \"implementation_approach\": {
        \"technical_strategy\": \"[High-level implementation approach]\",
        \"key_decisions\": [\"[Important technical decisions]\"],
        \"recommended_patterns\": [\"[Coding patterns to use]\"],
        \"frameworks_libraries\": [\"[Specific tools needed]\"]
      },
      \"acceptance_criteria\": [
        {
          \"criterion_id\": \"AC-T001-001\",
          \"description\": \"Given [context], when [action], then [expected result]\",
          \"verification_method\": \"[How to verify completion]\",
          \"complexity_level\": \"[simple/moderate/complex]\"
        }
      ],
      \"technical_requirements\": {
        \"files_to_create\": [
          {
            \"file_path\": \"[src/components/ComplexWizard.tsx]\",
            \"file_purpose\": \"[Multi-step wizard component]\",
            \"complexity_indicators\": [\"[State management, API integration]\"],
            \"key_exports\": [\"[Wizard, WizardStep, WizardProvider]\"],
            \"dependencies\": [\"[React, Redux, API client]\"]
          }
        ],
        \"files_to_modify\": [
          {
            \"file_path\": \"[src/api/wizardApi.ts]\",
            \"modification_scope\": \"[Add wizard endpoints]\",
            \"complexity_factors\": [\"[Multiple endpoint types, error handling]\"]
          }
        ],
        \"integration_points\": [
          {
            \"integration_name\": \"[AI Service Integration]\",
            \"complexity_contribution\": \"[High - custom AI workflow]\",
            \"implementation_notes\": \"[Specific integration requirements]\"
          }
        ]
      },
      \"subtasks\": [
        {
          \"subtask_id\": \"TASK-001-A\",
          \"subtask_title\": \"[Foundation: Basic Wizard Structure]\",
          \"subtask_scope\": \"[Create basic wizard framework without complex logic]\",
          \"subtask_rationale\": \"[Establish foundation for complex features]\",
          \"effort_estimate\": \"[Estimate for subtask]\",
          \"prerequisite_subtasks\": [],
          \"enables_subtasks\": [\"TASK-001-B\", \"TASK-001-C\"]
        },
        {
          \"subtask_id\": \"TASK-001-B\",
          \"subtask_title\": \"[AI Integration: Smart Recommendations]\",
          \"subtask_scope\": \"[Integrate AI service for component recommendations]\",
          \"subtask_rationale\": \"[Complex AI logic separated from UI concerns]\",
          \"effort_estimate\": \"[Estimate for subtask]\",
          \"prerequisite_subtasks\": [\"TASK-001-A\"],
          \"enables_subtasks\": [\"TASK-001-D\"]
        },
        {
          \"subtask_id\": \"TASK-001-C\",
          \"subtask_title\": \"[Data Management: Wizard State]\",
          \"subtask_scope\": \"[Implement complex state management for wizard data]\",
          \"subtask_rationale\": \"[State complexity requires focused implementation]\",
          \"effort_estimate\": \"[Estimate for subtask]\",
          \"prerequisite_subtasks\": [\"TASK-001-A\"],
          \"enables_subtasks\": [\"TASK-001-D\"]
        },
        {
          \"subtask_id\": \"TASK-001-D\",
          \"subtask_title\": \"[Integration: Complete Wizard Workflow]\",
          \"subtask_scope\": \"[Integrate all components into complete workflow]\",
          \"subtask_rationale\": \"[Final integration of complex components]\",
          \"effort_estimate\": \"[Estimate for subtask]\",
          \"prerequisite_subtasks\": [\"TASK-001-B\", \"TASK-001-C\"],
          \"enables_subtasks\": []
        }
      ],
      \"testing_strategy\": {
        \"testing_approach\": \"[single_task_testing/incremental_subtask_testing]\",
        \"unit_tests\": [
          {
            \"test_scope\": \"[Component behavior testing]\",
            \"test_complexity\": \"[simple/moderate/complex]\",
            \"test_files\": [\"[Wizard.test.tsx]\"]
          }
        ],
        \"integration_tests\": [
          {
            \"test_scenario\": \"[End-to-end wizard completion]\",
            \"complexity_factors\": [\"[AI integration, state persistence]\"],
            \"test_dependencies\": [\"[Mock AI service, test database]\"]
          }
        ]
      },
      \"definition_of_done\": [
        \"[All acceptance criteria verified]\",
        \"[Code review completed]\",
        \"[Unit tests passing with [X]% coverage]\",
        \"[Integration tests passing]\",
        \"[Performance requirements met]\",
        \"[UI matches design specifications]\",
        \"[Accessibility requirements validated]\",
        \"[Documentation updated]\"
      ],
      \"risk_factors\": [
        {
          \"risk_type\": \"[complexity/dependency/technical]\",
          \"risk_description\": \"[AI service integration complexity]\",
          \"probability\": \"[High/Medium/Low]\",
          \"impact\": \"[High/Medium/Low]\",
          \"mitigation_strategy\": \"[Incremental subtask approach reduces risk]\"
        }
      ]
    }
  ],
  \"task_relationships\": [
    {
      \"relationship_type\": \"[blocks/enables/integrates_with]\",
      \"source_task\": \"TASK-001\",
      \"target_task\": \"TASK-002\",
      \"dependency_strength\": \"[hard/soft]\",
      \"dependency_reason\": \"[Why this dependency exists]\"
    }
  ],
  \"implementation_phases\": [
    {
      \"phase_name\": \"[Phase 1: Foundation Tasks]\",
      \"phase_description\": \"[Simple tasks and foundations for complex features]\",
      \"included_tasks\": [
        \"[TASK-001: Setup and simple components]\",
        \"[TASK-003: Basic configurations]\"
      ],
      \"complexity_range\": \"[1-[threshold]]\",
      \"estimated_duration\": \"[Phase duration estimate]\",
      \"parallel_execution\": [
        {
          \"track_name\": \"[UI Foundation Track]\",
          \"tasks\": [\"[Simple UI tasks that can run in parallel]\"]
        },
        {
          \"track_name\": \"[Backend Foundation Track]\",
          \"tasks\": [\"[Simple backend tasks that can run in parallel]\"]
        }
      ]
    },
    {
      \"phase_name\": \"[Phase 2: Complex Feature Implementation]\",
      \"phase_description\": \"[Complex features broken into manageable subtasks]\",
      \"included_tasks\": [
        \"[TASK-002: Complex AI integration (broken into subtasks)]\",
        \"[TASK-004: Advanced workflow (broken into subtasks)]\"
      ],
      \"complexity_range\": \"[[threshold+1]-10]\",
      \"estimated_duration\": \"[Phase duration estimate]\",
      \"subtask_coordination\": [
        {
          \"feature_name\": \"[AI Project Wizard]\",
          \"subtask_sequence\": [\"[Foundation → AI Logic → State Management → Integration]\"],
          \"coordination_notes\": \"[How subtasks build on each other]\"
        }
      ]
    }
  ],
  \"effort_summary\": {
    \"total_estimate\": \"[Total effort across all tasks]\",
    \"simple_tasks_effort\": \"[Effort for tasks ≤ threshold]\",
    \"complex_tasks_effort\": \"[Effort for tasks > threshold]\",
    \"breakdown_efficiency\": \"[% effort saved by targeted breakdown]\",
    \"effort_by_complexity\": {
      \"low_complexity_1_3\": \"[Effort for score 1-3]\",
      \"medium_complexity_4_6\": \"[Effort for score 4-6]\",
      \"high_complexity_7_10\": \"[Effort for score 7-10]\"
    },
    \"implementation_timeline\": {
      \"sequential_estimate\": \"[If all tasks done sequentially]\",
      \"parallel_estimate\": \"[With optimal parallelization]\",
      \"team_size_recommendation\": \"[Recommended team composition]\"
    }
  },
  \"quality_gates\": [
    {
      \"gate_name\": \"[Simple Task Completion Gate]\",
      \"applicable_tasks\": \"[Tasks ≤ complexity threshold]\",
      \"criteria\": [
        \"[Task implemented according to acceptance criteria]\",
        \"[Code quality standards met]\",
        \"[Testing requirements satisfied]\"
      ]
    },
    {
      \"gate_name\": \"[Complex Feature Integration Gate]\",
      \"applicable_tasks\": \"[Tasks > complexity threshold]\",
      \"criteria\": [
        \"[All subtasks completed and integrated]\",
        \"[End-to-end functionality verified]\",
        \"[Performance requirements met]\",
        \"[Complex feature acceptance criteria satisfied]\"
      ]
    }
  ]
}
```

### 5. Adaptive Breakdown Logic

#### Threshold Decision Making:
```
IF complexity_score ≤ threshold:
    CREATE comprehensive single task with:
    - Complete implementation details
    - Full acceptance criteria
    - All technical specifications
    - Ready for immediate development
    
ELSE IF complexity_score > threshold:
    ANALYZE breakdown strategy:
    - Identify natural separation points
    - Create foundation subtasks
    - Separate complex logic into focused subtasks
    - Create integration subtasks
    - Maintain clear dependency chain
```

#### Subtask Creation Strategy:
- **Foundation First**: Basic structure and setup
- **Core Logic Separation**: Complex algorithms/AI in dedicated subtasks
- **Integration Last**: Bringing components together
- **Testing Throughout**: Each subtask has clear testing requirements

### 6. Quality Assurance and Validation
- **Completeness Check**: All user stories covered
- **Complexity Justification**: All scoring decisions explained
- **Dependency Validation**: Subtask dependencies are logical
- **Implementation Readiness**: Both simple and complex tasks are actionable

### 7. Output Generation and Summary
- Save as `adaptive_task_list.json`
- Create complexity analysis summary
- Generate implementation timeline recommendations
- Provide team allocation suggestions based on task complexity

## Adaptive Strategy Benefits:

### For Simple Tasks (≤ Threshold):
- **Efficiency**: No unnecessary breakdown overhead
- **Context Preservation**: All details in one place
- **Quick Implementation**: Developers can start immediately
- **Reduced Coordination**: No subtask dependencies to manage

### For Complex Tasks (> Threshold):
- **Risk Reduction**: Break complexity into manageable pieces
- **Parallel Development**: Multiple developers can work on subtasks
- **Clear Progress**: Incremental completion tracking
- **Quality Focus**: Dedicated attention to complex logic

## Example BOMthrower Task Scoring:

**Simple Tasks (1-3):**
- \"Create project structure and package.json\" - Score: 2
- \"Add basic Button component\" - Score: 2
- \"Set up routing between main pages\" - Score: 3

**Medium Tasks (4-6):**
- \"Implement user authentication flow\" - Score: 5
- \"Create component database schema\" - Score: 4
- \"Add file upload functionality\" - Score: 6

**Complex Tasks (7-10):**
- \"Build AI Project Wizard with smart recommendations\" - Score: 9
- \"Implement Smart Component Matcher algorithm\" - Score: 8
- \"Create Budget Guardian with real-time pricing\" - Score: 8

**With Threshold = 6:**
- Simple and medium tasks: Single comprehensive tasks
- Complex tasks: Broken into foundation → core logic → integration subtasks

## Success Criteria:
- All user stories converted to appropriately scoped tasks
- Tasks ≤ threshold are implementation-ready single tasks
- Tasks > threshold are broken into logical, dependency-mapped subtasks
- Effort estimates reflect actual implementation complexity
- Development teams can execute tasks without further breakdown

The adaptive task breakdown is complete and optimized for your team's implementation approach."
```