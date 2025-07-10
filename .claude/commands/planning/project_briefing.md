# Project Briefing Command

```bash
claude-code "Get up to speed on the current project by reviewing all essential context and understanding the current development state.

## Task: Load Project Context and Understanding

**Purpose:** Provide comprehensive briefing on project status, architecture, requirements, and development approach to enable effective task orchestration.

## Context Loading Process:

### 1. Project Overview and Workflow
- Read `claude.md` - Project description, AI workflow guide, and development approach
- Understand the manifest-driven development methodology
- Learn the available commands and when to use them

### 2. Current Implementation State
- Read `codebase_manifest.json` - Current state of all implemented files and components
- Understand what has been built so far
- Identify current project phase and development status

### 3. Target Architecture
- Read `docs/proposed_final_manifest.json` - Complete planned architecture
- Understand the target system design and component relationships
- Compare current state vs. planned final state

### 4. Requirements and Specifications
- Read `docs/mvp.md` - Core feature requirements and project goals
- Read `docs/prd.md` (if exists) - Detailed product requirements
- Understand user needs and acceptance criteria

### 5. Implementation Plan
- Read `tasks/task_list.md` - Complete development task breakdown
- Understand task dependencies and current progress
- Identify next tasks ready for implementation

### 6. Architectural Evolution
- Read `docs/manifest_evolution.md` - Track architectural decisions and learnings
- Understand how the project has evolved during implementation
- Note any important design decisions or changes

### 7. Recent Progress Assessment
- Check `tasks/completed/` directory for recently completed tasks
- Review the latest 2-3 completed task files to understand recent work
- Check any validation reports to understand implementation quality

### 8. Development Environment Status
- Review project structure and build configuration
- Understand current dependencies and tooling
- Assess readiness for continued development

## Summary Generation:

After reviewing all context, provide a comprehensive project briefing:

### Project Summary:
- **Project Name:** [Name and description]
- **Technology Stack:** [Main technologies used]
- **Current Phase:** [Which development phase we're in]
- **Development Status:** [What's been completed, what's next]

### Architecture Overview:
- **Main Components:** [Key system components and their roles]
- **Data Flow:** [How information flows through the system]
- **Integration Points:** [External systems and dependencies]
- **Current vs Target:** [How far along we are toward final architecture]

### Implementation Progress:
- **Completed Tasks:** [Recently finished work]
- **Current Phase:** [What phase we're working on]
- **Next Tasks:** [What's ready to be implemented]
- **Dependencies:** [Any blockers or prerequisites]

### Development Approach:
- **Workflow:** [Manifest-driven process overview]
- **Available Commands:** [Key commands and when to use them]
- **Quality Assurance:** [Validation and testing approach]
- **Documentation:** [How architecture and decisions are tracked]

### Ready for Task Orchestration:
- **Task Status:** [Current task completion status]
- **Next Task Recommendations:** [What should be worked on next]
- **Context Loaded:** [Confirm all necessary context is understood]
- **Ready to Proceed:** [Confirmation that orchestration can begin]

## Key Context Files to Review:

**Essential (always read):**
- `claude.md` - Project workflow and AI instructions
- `codebase_manifest.json` - Current implementation state
- `docs/proposed_final_manifest.json` - Target architecture
- `docs/mvp.md` - Core requirements
- `tasks/task_list.md` - Implementation plan

**Important (read if exists):**
- `docs/prd.md` - Detailed requirements
- `docs/manifest_evolution.md` - Architectural decisions
- `tasks/completed/[latest-tasks].json` - Recent progress
- `tasks/validation/[recent-validation].json` - Quality reports

**Supplementary (scan if time permits):**
- `README.md` - Project overview
- `package.json` - Dependencies and scripts
- `docs/` directory contents - Additional documentation

## Context Efficiency Guidelines:

### Prioritize Information:
1. **Critical:** Current state, target architecture, next tasks
2. **Important:** Requirements, recent progress, development approach
3. **Helpful:** Historical decisions, validation reports, supplementary docs

### Focus Areas:
- What has been built and what remains
- Current development phase and next logical steps
- Any architectural decisions or changes from implementation
- Task dependencies and readiness for orchestration

### Output Format:
- Concise but comprehensive summary
- Clear current status and next steps
- Confirmation of readiness for task orchestration
- Specific recommendations for next actions

## Success Criteria:
- Complete understanding of project goals and architecture
- Clear picture of current implementation state
- Understanding of manifest-driven development approach
- Knowledge of available commands and workflow
- Readiness to effectively orchestrate tasks
- Specific recommendations for next development steps

After this briefing, you should be fully prepared to:
- Use orchestrate_task commands effectively
- Understand task dependencies and sequencing
- Make informed decisions about implementation approach
- Provide guidance on project development strategy

The project context is now loaded and ready for effective development orchestration."
```