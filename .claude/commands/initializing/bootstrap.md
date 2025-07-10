# Project Bootstrap Prompt

Use this prompt with Claude Code to bootstrap a project structure from existing planning documents:

```bash
claude-code "Bootstrap this project using manifest-driven development. I have the following planning documents and command files in the current directory that need to be organized:

**Required Files (must be present):**
- MVP/PRD documents  
- Task list document
- AI manifest workflow document
- generate_manifest.md (command prompt)
- process_task.md (command prompt)
- implement_task.md (command prompt)
- check_task.md (command prompt)
- resolve_mismatch.md (command prompt)
- commit_task.md (command prompt)
- update_final_manifest.md (command prompt)
- Sample manifest JSON (optional)
- Any other planning documents

**Expected Filenames:**
- `mvp.md` - MVP document
- `prd.md` - PRD document (if exists)
- `task_list.md` - Task list document
- `ai_workflow.md` - AI manifest workflow document
- `proposed_final_manifest.json` - Sample manifest JSON
- Command prompts with exact names listed above

## Task: Organize Project Structure

### 1. Create Directory Structure
```
project-root/
├── docs/
│   ├── mvp.md (move MVP document here)
│   ├── prd.md (move PRD document here if exists)
│   ├── proposed_final_manifest.json (move sample manifest here)
│   └── manifest_evolution.md (create - for tracking manifest updates)
├── .claude/
│   └── commands/
│       ├── generate_manifest.md (move generate manifest prompt here)
│       ├── process_task.md (create this - see below)
│       ├── implement_task.md (create this - see below)
│       ├── check_task.md (create this - see below)
│       └── resolve_mismatch.md (create this - see below)
├── tasks/
│   ├── task_list.md (move task list here)
│   ├── prepared/ (for processed task files)
│   └── completed/ (for finished task files)
├── .gitignore
├── README.md
├── claude.md (create from AI workflow document)
└── codebase_manifest.json (create empty structure)
```

### 2. Create .claude/commands/ directory and move existing command files:
- generate_manifest.md (move existing file here)
- process_task.md (move existing file here)
- implement_task.md (move existing file here)
- check_task.md (move existing file here)
- resolve_mismatch.md (move existing file here)
- commit_task.md (move existing file here)
- update_final_manifest.md (move existing file here)

### 3. Create claude.md
Use the AI manifest workflow document to create claude.md. Include:

**Core Content:**
- Overview of manifest-driven development
- Reference to docs/ directory files (MVP, PRD, sample manifest)
- Reference to tasks/ directory structure
- Complete development workflow steps

**Command Overview Section:**
- Available commands in .claude/commands/:
  - `generate_manifest.md` - Analyze codebase and create/update manifests
  - `process_task.md` - Prepare tasks with expected post-task manifests
  - `implement_task.md` - Implement prepared tasks with full context
  - `check_task.md` - Validate implementation against expected manifest
  - `resolve_mismatch.md` - Handle discrepancies between expected and actual
  - `commit_task.md` - Commit completed tasks with proper git history
  - `update_final_manifest.md` - Update proposed final manifest based on learnings
- How commands work together in the workflow
- Examples of command usage
- When to use each command
- Git workflow integration

**MCP Servers and Tools Section:**
- List of available MCP servers and tools
- Rules for when to use MCP tools vs. built-in commands
- Integration guidelines for MCP tools
- **Note:** This section to be expanded later with specific MCP tool details

**Future Enhancements Section:**
- TODO: Move commands to global ~/.claude/commands for reuse across projects
- TODO: Tasks management system (possibly integrate taskmaster-ai or similar MCP tool)
- TODO: Artifact generation integration
- TODO: Expand MCP tools integration and usage guidelines

**Development Workflow Section:**
- Step-by-step process using the commands
- Examples of complete task implementation cycle
- Best practices for manifest-driven development
- Troubleshooting common issues

### 8. Create .gitignore
```
# Dependencies
node_modules/

# Build outputs
dist/
build/

# Environment files
.env
.env.local

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*

# Task working files
tasks/prepared/  # Temporary task files with full context
!tasks/prepared/.gitkeep  # Keep directory structure
```

### 9. Create README.md
```markdown
# [Project Name from MVP/PRD]

[Brief description from MVP/PRD]

## Project Overview

- **Tech Stack:** [Extract from MVP/PRD]
- **Type:** [e.g., Obsidian Plugin, Web App, CLI Tool]
- **Development Status:** Initial setup complete, ready for implementation

## Development Approach

This project uses manifest-driven development with task-by-task implementation. The `codebase_manifest.json` file contains complete project information including:

- Project metadata and tech stack
- Documentation references
- Architecture overview from planning documents
- Development workflow tracking

See `claude.md` for detailed AI workflow instructions.

## Quick Start

1. Review `docs/mvp.md` for project requirements
2. Check `docs/prd.md` for detailed specifications (if available)
3. Review `tasks/task-list.md` for implementation plan
4. Check `codebase_manifest.json` for current project state
5. Follow development workflow in `claude.md`

## Directory Structure

- `docs/` - Project documentation (MVP, PRD, sample manifest)
- `tasks/` - Development task list and task processing
- `.claude/commands/` - AI command prompts for development workflow
- `codebase_manifest.json` - Complete project manifest with metadata
- `claude.md` - AI workflow documentation

## Development Workflow

1. `claude-code process_task \"Task-X.X\"` - Prepare task with expected manifest
2. `claude-code implement_task \"tasks/prepared/Task-X.X.json\"` - Implement changes
3. `claude-code check_task \"Task-X.X\"` - Verify implementation matches expected
4. If mismatch: `claude-code resolve_mismatch \"Task-X.X\"` - Handle discrepancies

## Project Status

The project is bootstrapped and ready for implementation. The manifest contains project information extracted from planning documents and will be updated as development progresses.

See `claude.md` for detailed instructions and `codebase_manifest.json` for current project state.
```

### 10. Create codebase_manifest.json
```json
{
  \"version\": \"1.0\",
  \"generated\": \"[current timestamp]\",
  \"project\": {
    \"name\": \"[infer from directory name or MVP/PRD]\",
    \"description\": \"[extract from MVP/PRD description]\",
    \"version\": \"0.1.0\",
    \"tech_stack\": \"[infer from MVP/PRD - e.g., TypeScript, Node.js, Obsidian Plugin]\",
    \"deployment\": \"[extract deployment info from MVP/PRD if mentioned]\",
    \"repository\": \"[to be added when remote is configured]\"
  },
  \"documentation\": {
    \"mvp\": \"docs/mvp.md\",
    \"prd\": \"docs/prd.md\",
    \"task_list\": \"tasks/task_list.md\",
    \"proposed_final_manifest\": \"docs/proposed_final_manifest.json\",
    \"manifest_evolution\": \"docs/manifest_evolution.md\",
    \"architecture_notes\": \"[extract high-level architecture from MVP/PRD]\"
  },
  \"files\": {
    \"// Note\": \"Files will be added as they are implemented through tasks\"
  },
  \"dependencies\": {
    \"// Note\": \"Dependencies will be added based on tech stack and implementation needs\"
  },
  \"architecture\": {
    \"main_flow\": \"[extract from MVP/PRD if described]\",
    \"data_flow\": \"[extract from MVP/PRD if described]\",
    \"configuration\": \"[extract configuration approach from MVP/PRD]\",
    \"key_components\": \"[list main components mentioned in MVP/PRD]\",
    \"integration_points\": \"[list external systems mentioned in MVP/PRD]\"
  },
  \"development\": {
    \"approach\": \"manifest-driven development with git workflow integration\",
    \"workflow\": \"process_task -> implement_task -> check_task -> resolve_mismatch (if needed) -> commit_task\",
    \"task_status\": \"ready to begin - see tasks/task_list.md\",
    \"current_phase\": \"initial setup\",
    \"manifest_evolution\": \"tracked in docs/manifest_evolution.md\",
    \"version_control\": \"git commits tied to task completion\"
  }
}
```

## Instructions for creating the manifest:
1. **Read MVP/PRD documents** to extract project information
2. **Infer tech stack** from requirements (e.g., if mentions Obsidian plugin, include TypeScript/Node.js)
3. **Extract architecture details** from any technical sections
4. **Note configuration approach** (settings files, environment variables, etc.)
5. **List key components** mentioned in planning documents
6. **Identify integration points** (APIs, external services, etc.)
7. **Set initial development status** to track progress

### 11. Create docs/manifest_evolution.md
```markdown
# Manifest Evolution Log

This document tracks changes to the proposed final manifest as the project evolves.

## Initial Version - [Date]

### Source
Created from initial project planning and requirements analysis.

### Key Components
- [List initial key components from MVP/PRD]

### Architecture Decisions
- [List initial architectural decisions]

### Future Updates
Updates will be logged here as the project evolves and we learn from implementation.
```
- `tasks/prepared/` - Temporary task files with full context (gitignored)
- `tasks/completed/` - Completed task records for project history (tracked in git)
- Add `.gitkeep` files to ensure directories are created

### 12. Initialize Git
```bash
git init
git add .
git commit -m \"Initial project bootstrap with manifest-driven development structure\"
```

## Final Steps:
1. **Verify all required files are present** in the current directory
2. **Move all existing documents** to appropriate directories
3. **Create all new files** as specified above
4. **Initialize git repository** and create initial commit
5. **Verify project structure** is complete and ready for development

## Required Files Before Bootstrap:
- MVP/PRD documents
- Task list document
- AI manifest workflow document  
- generate_manifest.md
- process_task.md
- implement_task.md
- check_task.md
- resolve_mismatch.md
- commit_task.md
- update_final_manifest.md
- Sample manifest JSON (optional)

**Expected Filenames:**
- `mvp.md`
- `prd.md` 
- `task_list.md`
- `ai_workflow.md`
- `proposed_final_manifest.json`
- All command .md files with exact names listed above

The project is now ready for the task-by-task implementation workflow."
```