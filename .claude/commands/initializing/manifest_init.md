# Manifest Init Command

```bash
claude-code "Initialize an existing project for manifest-driven development workflow.

## Task: Initialize Manifest-Driven Development

**Project Type:** [Auto-detect from package.json, requirements.txt, etc.]

Start your response with: "🚀 **MANIFEST_INIT EXECUTING** - Transitioning existing project to manifest-driven development"

## Initialization Process:

### 1. Project Analysis and Detection
- **Scan project structure** to understand current organization
- **Detect project type**: React, Node.js, Python, Java, etc.
- **Identify existing documentation**: README.md, docs/, PRDs, etc.
- **Find configuration files**: package.json, requirements.txt, Cargo.toml, etc.
- **Detect version control**: Git status, existing branches, commit history
- **Identify deployment setup**: Vercel, Netlify, Docker, CI/CD files

### 2. Create Manifest-Driven Structure
Create the essential directory structure:
```
project-root/
├── .claude/
│   └── commands/
│       ├── generate_manifest.md
│       ├── process_task.md
│       ├── implement_task.md
│       ├── check_task.md
│       ├── resolve_mismatch.md
│       ├── commit_task.md
│       └── update_final_manifest.md
├── docs/
│   ├── current_state.md (NEW)
│   ├── proposed_final_manifest.json (NEW)
│   ├── manifest_evolution.md (NEW)
│   └── [existing docs moved here]
├── tasks/
│   ├── task_list.md (NEW)
│   ├── prepared/ (for processed task files)
│   ├── completed/ (for finished task files)
│   └── validation/ (for task validation reports)
└── codebase_manifest.json (NEW)
```

### 3. Install Command System via GitHub Repository
Install all proven commands from your AIhelpers repository:

```bash
# Clone commands from proven repository
git clone --depth 1 https://github.com/loydmilligan/AIhelpers.git .tmp-aihelpers
mkdir -p .claude/commands
cp .tmp-aihelpers/prompts/Claude/commands/*.md .claude/commands/
rm -rf .tmp-aihelpers
```

**Core Development Commands Installed:**
- **bootstrap.md** - Bootstrap new projects with manifest-driven structure
- **generate_manifest.md** - Analyze codebase and create/update manifests
- **process_task.md** - Prepare tasks with expected post-task manifests
- **implement_task.md** - Implement prepared tasks with full context
- **check_task.md** - Validate implementation against expected manifest
- **resolve_mismatch.md** - Handle discrepancies between expected and actual
- **commit_task.md** - Commit completed tasks with proper git history
- **update_final_manifest.md** - Update proposed final manifest based on learnings

**Workflow Management Commands:**
- **orchestrate_task.md** - Complete task lifecycle management with sub-agents
- **project_briefing.md** - Comprehensive project context loading and briefing

**Intelligence Gathering Commands:**
- **users.md** - Multi-agent user research and persona development (4-10 agents)
- **market.md** - Multi-agent market research and competitive analysis (4-12 agents)
- **features.md** - Multi-agent feature ideation and suggestion system (5-12 agents)

**Benefits of GitHub Approach:**
- Always up-to-date with latest command improvements
- Centralized command management across projects
- Easy to update commands globally when enhanced
- Proven, battle-tested command set from your existing workflow

### 4. Generate Initial Documentation

#### 4.1 Create Current State Analysis (`docs/current_state.md`)
```markdown
# Project Current State Analysis

**Generated:** [timestamp]
**Project Type:** [detected type]
**Codebase Size:** [file count, LOC estimate]

## Architecture Overview
[Detected from file structure and dependencies]

## Key Components
[Main modules, services, components identified]

## Dependencies
[Major dependencies and their purposes]

## Deployment Status
[Production/staging environments detected]

## Technical Debt Areas
[Potential improvement areas identified]

## Documentation Status
[Existing docs found and their quality]
```

#### 4.2 Generate Current Manifest via Sub-Agent
Deploy manifest generation sub-agent using the proven generate_json_manifest.md command:

**Sub-Agent Task: Generate Current Manifest**
```
You are a code analysis agent. Your ONLY task is to analyze the source code of a project and generate a detailed JSON manifest. You MUST NOT summarize the project. You MUST follow the JSON schema provided.

## Task: Generate codebase_manifest.json

Start your response with: "🔍 **GENERATE_MANIFEST EXECUTING** - Performing detailed code analysis and creating codebase_manifest.json"

## CRITICAL INSTRUCTIONS:
1. **OUTPUT FORMAT IS JSON:** The final output MUST be a single, valid JSON file named `codebase_manifest.json`. Do NOT output Markdown or any other format.
2. **PERFORM DEEP CODE ANALYSIS:** You MUST scan every source file (`.js`, `.jsx`, `.ts`, `.tsx`, `.py`, etc.) individually. Do NOT rely on `package.json` or `README.md` for the file analysis section.
3. **DO NOT SUMMARIZE:** Your task is to create a structured representation of the code, not a human-readable summary. The `files` object in the JSON must contain an entry for every single source file.
4. **ADHERE TO THE SCHEMA:** The generated JSON must strictly follow the schema defined below. Every required field must be present.

## Manifest JSON Schema:

```json
{
  "version": "1.0",
  "generated": "[current timestamp in ISO format]",
  "project": "[infer project name from package.json or directory]",
  "description": "[brief description from README or package.json]",
  "files": {
    "path/to/file.ext": {
      "purpose": "[one-line, inferred description of this file's specific role]",
      "exports": {
        "functions": [
          {
            "name": "function_name",
            "signature": "function_name(param1: type, param2: type) -> return_type",
            "description": "what this function does",
            "parameters": {
              "param1": "description of param1",
              "param2": "description of param2"
            },
            "returns": "description of return value"
          }
        ],
        "classes": [
          {
            "name": "ClassName",
            "description": "what this class does",
            "constructor": "ClassName(param1: type, param2: type)",
            "methods": [
              {
                "name": "method_name",
                "signature": "method_name(param: type) -> return_type",
                "description": "what this method does",
                "parameters": {"param": "description"},
                "returns": "description of return value"
              }
            ],
            "properties": [
              {
                "name": "property_name",
                "type": "property_type",
                "description": "what this property stores"
              }
            ]
          }
        ],
        "constants": [
          {
            "name": "CONSTANT_NAME",
            "type": "constant_type",
            "value": "actual_value_or_description",
            "description": "what this constant represents"
          }
        ]
      },
      "imports": ["list of all imported modules and local files"],
      "sideEffects": ["list of side effects like 'writes-database', 'network-calls', etc."]
    }
  },
  "dependencies": {
    "[package-name]": "[brief description of what this dependency provides]"
  },
  "architecture": {
    "main_flow": "[describe the main execution flow, inferred from code]",
    "data_flow": "[describe how data flows through the system, inferred from code]",
    "configuration": "[describe how the system is configured]"
  }
}
```

## Analysis Instructions:
1. Scan all source code files in the current directory and subdirectories.
2. Ignore these files/directories: node_modules/, .git/, dist/, build/, .DS_Store, *.log files, .env files (but note if they exist in the configuration section)
3. For each file, you MUST determine: Purpose, Exports (detailed list of all functions, classes, and constants), Imports, Side Effects.
4. For exports, provide complete API documentation as per the schema.
5. Side Effects Categories: 'writes-database', 'reads-database', 'network-calls', 'sends-data', 'receives-data', 'publishes-events', 'subscribes-to-events', 'writes-files', 'reads-files', 'creates-ui', 'modifies-dom', 'registers-events', 'registers-commands', 'loads-settings', 'saves-settings'
6. For package.json dependencies, read the actual dependencies and provide brief descriptions.
7. Architecture Analysis: Infer the architecture from the code itself, trace the primary execution flow from the entry point.

## Final Output Requirement:
Create the file named codebase_manifest.json. The file's content must be only the JSON object described above.
```

This uses your proven generate_json_manifest.md command to create our baseline understanding of the existing codebase.

#### 4.3 Defer Proposed Final Manifest Creation
**HOLD**: Do not create proposed final manifest yet. This requires intelligence gathering first.
- Current task: Document baseline understanding only
- Future task: Create proposed manifest after task identification phase

#### 4.4 Create Evolution Log (`docs/manifest_evolution.md`)
```markdown
# Manifest Evolution Log

## Initial Version - [Date]

### Source
Created during manifest-driven development initialization from existing project.

### Current State
[Summary of what exists]

### Planned Improvements
[Based on detected improvement areas]

### Future Updates
Updates will be logged here as the project evolves through manifest-driven development.
```

#### 4.5 Create Intelligence Gathering Task List (`tasks/task_list.md`)
Generate initial intelligence-gathering tasks focused on understanding project direction:

**Phase 1: Project Intelligence Tasks**
- **Task-INT-1**: Identify and catalog existing roadmap/PRD documents
- **Task-INT-2**: Analyze unfinished or partially completed features  
- **Task-INT-3**: Schedule stakeholder brainstorming session on future plans
- **Task-INT-4**: Generate user feedback survey and prioritization framework
- **Task-INT-5**: Perform codebase health assessment (refactoring, testing, type safety)
- **Task-INT-6**: Deploy sub-agent orchestration for comprehensive analysis:
  
  **Sub-Agent Orchestration Tasks:**
  ```bash
  # Deploy user research agents (4-10 specialized agents)
  claude-code users.md "Project Description" "comprehensive" "research/users/"
  
  # Deploy market research agents (4-12 specialized agents) 
  claude-code market.md "Project Description" "deep" "research/market/"
  
  # Deploy feature ideation agents (5-12 specialized agents)
  claude-code features.md "Project Description" "8" "research/features/"
  ```
  
  These orchestration commands use the full command content from your AIhelpers repo to deploy multiple specialized research agents simultaneously, providing comprehensive intelligence across user needs, market opportunities, and feature possibilities.

**Phase 2: Strategic Planning Tasks** (Created after Phase 1 completion)
- **Task-PLAN-1**: Synthesize intelligence into strategic priorities
- **Task-PLAN-2**: Create proposed final manifest based on gathered intelligence
- **Task-PLAN-3**: Generate implementation task breakdown structure

**Intelligence Gathering Principles:**
- Focus on understanding before building ("Bad in, bad out")
- Use proven multi-agent orchestration for comprehensive analysis
- Defer implementation planning until intelligence is complete
- Leverage specialized research agents for deep domain expertise

### 5. Project Integration

#### 5.1 Update .gitignore
Add manifest-driven workflow entries:
```
# Manifest-driven development
tasks/prepared/          # Temporary task files with full context
tasks/validation/        # Task validation reports (optional)
!tasks/prepared/.gitkeep
!tasks/validation/.gitkeep
```

#### 5.2 Update/Create README.md
Enhance existing README with manifest-driven workflow section:
```markdown
## Development Workflow

This project uses manifest-driven development for systematic feature implementation.

### Quick Start
1. Review `docs/current_state.md` for project overview
2. Check `docs/proposed_final_manifest.json` for planned architecture
3. See `tasks/task_list.md` for implementation roadmap
4. Follow development workflow in `.claude/commands/`

### Development Cycle
1. `claude-code process_task "Task-X.X"` - Prepare task
2. `claude-code implement_task "tasks/prepared/Task-X.X.json"` - Implement
3. `claude-code check_task "Task-X.X"` - Validate
4. `claude-code commit_task "Task-X.X"` - Commit

See `.claude/commands/` for detailed command documentation.
```

#### 5.3 Create claude.md from Proven ai_workflow.md Template
Create comprehensive development workflow documentation based on your proven ai_workflow.md:

**Source Template**: Use ai_workflow.md as the foundation - it's battle-tested and comprehensive
**Customization**: Adapt content for this specific project while preserving proven patterns

**Content Structure (from ai_workflow.md):**
- Manifest-driven approach explanation
- Example workflows with step-by-step AI processes  
- Validation examples and manifest update patterns
- Complete development workflow cycle (process → implement → check → resolve → commit)
- Command usage examples and when to use each
- Project structure documentation
- Request format examples for optimal AI interaction
- Project evolution and architectural decision tracking

**Project-Specific Additions:**
- Current project technology stack and architecture
- Intelligence gathering task descriptions (Task-INT-1 through Task-INT-6)
- Integration with orchestration commands (users.md, market.md, features.md)
- Specific examples relevant to this project's domain

**Integration Strategy:**
- Preserve all proven workflow patterns from ai_workflow.md
- Add project-specific context and examples
- Maintain the comprehensive, reference-quality documentation style
- Include intelligence-first approach for task identification

This creates a complete workflow guide tailored to this project while leveraging your proven methodology.

### 6. Validation and Testing

#### 6.1 Test Command Installation and Manifest Generation
- Verify all commands downloaded successfully from AIhelpers GitHub repository
- Test generate_manifest.md command execution on the project
- Validate that all major components are captured in generated manifest
- Check for any parsing errors or missing files
- Ensure orchestration commands (users.md, market.md, features.md) are available and ready
- Verify workflow commands (process_task.md, implement_task.md, check_task.md, etc.) are installed

#### 6.2 Verify Project Structure and Documentation  
- Confirm directory structure created correctly (.claude/commands/, docs/, tasks/)
- Test that claude.md documentation is comprehensive and project-specific
- Validate current_state.md accurately reflects project characteristics
- Check that intelligence gathering task list references the correct command usage
- Ensure all commands from AIhelpers repo are accessible and ready for use

#### 6.3 Create Intelligence Gathering Sample Task
Generate a low-risk intelligence gathering task to test the workflow:
- **Task-INT-SAMPLE**: Test orchestration commands with small-scale analysis
- Validate that sub-agent orchestration works with installed commands
- Test the intelligence-first approach with a simple user research deployment
- Demonstrates the complete workflow cycle: process → implement → check → commit

### 7. Preserve Existing Workflow

#### 7.1 Non-Disruptive Integration
- **Preserve all existing files** in their current locations
- **Keep existing package.json, configs** unchanged
- **Maintain git history** and existing branches
- **Don't break existing CI/CD** or deployment processes

#### 7.2 Individual Developer Workflow
- Manifest-driven workflow is additive, not replacing existing practices
- Can adopt new workflow gradually at your own pace  
- Old and new approaches can coexist during transition
- Focus on proving value with low-risk tasks first

### 8. Project-Specific Customization

#### 8.1 Technology-Specific Enhancements
**For React Projects:**
- Identify component architecture patterns
- Detect state management approach (Redux, Context, etc.)
- Note build system (Vite, Create React App, etc.)
- Plan component modernization tasks

**For Node.js/Express:**
- Map API routes and middleware
- Identify database connections
- Document service integrations
- Plan refactoring opportunities

**For Python Projects:**
- Detect framework (Django, Flask, FastAPI)
- Map package structure
- Identify dependencies and requirements
- Plan testing improvements

#### 8.2 Domain-Specific Improvements
Based on project type, suggest relevant tasks:
- Performance optimization
- Security improvements  
- Testing coverage
- Documentation enhancement
- Code modernization

### 9. Success Criteria Validation

#### 9.1 Structural Validation
- [ ] All 7 commands properly installed
- [ ] Directory structure created correctly
- [ ] Current manifest generated successfully
- [ ] Documentation files created

#### 9.2 Functional Validation
- [ ] generate_manifest command works on project
- [ ] Sample task can be processed and implemented
- [ ] Git workflow integration functions properly
- [ ] Existing development workflow unaffected

#### 9.3 Quality Validation
- [ ] Current state accurately documented
- [ ] Proposed improvements are realistic
- [ ] Task list is actionable and prioritized
- [ ] Team can understand and adopt workflow

## Output Summary:

Report initialization results:
- **Project Analysis**: Type, size, key characteristics
- **Structure Created**: Directories and files added  
- **Commands Installed**: All 7 workflow commands + 3 intelligence gathering commands available
- **Current State Documented**: Baseline understanding established
- **Intelligence Tasks Generated**: Strategic research tasks ready for execution
- **Validation Status**: What works and any issues found
- **Next Steps**: Recommended intelligence gathering approach

## Example Next Steps:
1. **Review generated current state documentation** to ensure accuracy
2. **Begin intelligence gathering tasks** (Task-INT-1 through Task-INT-6)
3. **Use orchestration commands** for comprehensive research (users, market, features)
4. **Synthesize intelligence** before creating proposed final manifest
5. **Create strategic task list** based on gathered intelligence
6. **Begin first implementation task** using proven workflow

The project is now ready for intelligence-driven, systematic development planning while preserving all existing workflows and capabilities."
```