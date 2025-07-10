# Retrofit Existing Project for Manifest-Driven Development

Use this prompt with Claude Code to retrofit an existing project with manifest-driven development workflow:

```bash
claude-code "Retrofit this existing project for manifest-driven development. This is an existing codebase that needs the manifest-driven workflow added without disrupting current structure.

## Task: Add Manifest-Driven Development to Existing Project

### 1. Analyze Current Project Structure
- Scan current directory structure
- Identify existing documentation in `docs/` 
- Check for existing `.gitignore`, `README.md`, package files
- Note current git status (don't disrupt existing repository)

### 2. Create Missing Manifest-Driven Directories
Only create directories that don't already exist:

```
├── .claude/
│   └── commands/ (create if missing)
├── tasks/ (create if missing)
│   ├── prepared/ (create - for processed task files)
│   ├── completed/ (create - for finished task files)
│   └── validation/ (create - for validation reports)
```

### 3. Add Command Files to .claude/commands/
Create these command files in `.claude/commands/` if they don't exist:
- `generate_manifest.md`
- `process_task.md` 
- `implement_task.md`
- `check_task.md`
- `resolve_mismatch.md`
- `commit_task.md`
- `update_final_manifest.md`
- `orchestrate_task.md`

**Important**: Use the updated command content provided, not the bootstrap versions.

### 4. Create claude.md (Main AI Instructions)
Create `claude.md` in project root with sections:

**Project Context Section:**
- Brief description: \"This is the [PROJECT_NAME] - [extract description from existing docs]\"
- Technology stack: [analyze existing package.json, dependencies, file types]
- Current development status: \"Existing codebase retrofitted for manifest-driven development\"

**Available Commands Section:**
Document all commands in `.claude/commands/`:
```markdown
## Available Commands

### Core Workflow Commands
- \`generate_manifest\` - Create/update codebase_manifest.json from current state
- \`process_task\` - Prepare tasks with expected post-task manifests  
- \`implement_task\` - Implement prepared tasks with full context
- \`check_task\` - Validate implementation against expected manifest
- \`resolve_mismatch\` - Handle discrepancies between expected and actual
- \`commit_task\` - Commit completed tasks with proper git history
- \`orchestrate_task\` - Run complete workflow for a task

### Usage Examples
\`\`\`bash
# Generate current state manifest
claude-code \"generate_manifest\"

# Full task workflow  
claude-code \"orchestrate_task Task-2.1\"

# Individual steps
claude-code \"process_task Task-2.1\"
claude-code \"implement_task tasks/prepared/Task-2.1.json\"
claude-code \"check_task Task-2.1\"
\`\`\`
```

**Existing Documentation Section:**
- Reference existing docs in `docs/` directory
- Note location of task list: `tasks/task_list.md`
- Reference any existing architecture docs

**Development Workflow Section:**
- Task-by-task implementation approach
- Git integration with existing repository
- How to handle existing codebase during development

### 5. Generate Initial codebase_manifest.json
Create initial manifest by analyzing existing codebase:

```json
{
  \"version\": \"1.0\",
  \"generated\": \"[current timestamp]\",
  \"project\": {
    \"name\": \"[extract from package.json or directory name]\",
    \"description\": \"[extract from existing README or docs]\",
    \"version\": \"[extract from package.json or set to current]\",
    \"tech_stack\": \"[analyze existing dependencies and file types]\",
    \"deployment\": \"[extract from existing docs or deployment files]\",
    \"repository\": \"[extract from git remote if available]\"
  },
  \"documentation\": {
    \"readme\": \"README.md\",
    \"task_list\": \"tasks/task_list.md\",
    \"existing_docs\": \"[list existing files in docs/ directory]\",
    \"architecture_notes\": \"[reference existing architecture docs]\"
  },
  \"files\": {
    \"[analyze and document existing files - at least key entry points]\"
  },
  \"dependencies\": {
    \"[extract from package.json or requirements files]\"
  },
  \"architecture\": {
    \"main_flow\": \"[analyze existing codebase to determine main execution flow]\",
    \"data_flow\": \"[analyze existing data handling]\",
    \"configuration\": \"[analyze existing config approach]\",
    \"key_components\": \"[identify main modules/components]\",
    \"integration_points\": \"[identify external service integrations]\"
  },
  \"development\": {
    \"approach\": \"manifest-driven development retrofitted to existing codebase\",
    \"workflow\": \"task-by-task implementation with manifest validation\",
    \"current_phase\": \"retrofit complete - ready for task-based development\",
    \"version_control\": \"git commits tied to development milestones\"
  }
}
```

### 6. Create Initial Proposed Final Manifest
**If docs/proposed_final_manifest.json doesn't exist:**
Create target architecture manifest from existing documentation:

```json
{
  \"version\": \"1.0\",
  \"generated\": \"[current timestamp]\",
  \"status\": \"initial_target_architecture\",
  \"project\": {
    \"name\": \"[extract from package.json or directory name]\",
    \"description\": \"[extract from existing README or docs]\",
    \"version\": \"[target version - usually 1.0.0]\",
    \"tech_stack\": \"[analyze existing dependencies and planned architecture]\",
    \"deployment\": \"[extract from existing docs or analyze Docker files]\",
    \"repository\": \"[extract from git remote if available]\"
  },
  \"documentation\": {
    \"readme\": \"README.md\",
    \"task_list\": \"tasks/task_list.md\",
    \"existing_docs\": \"[list key documentation files found]\",
    \"architecture_notes\": \"[extract from existing architecture docs]\"
  },
  \"files\": {
    \"[analyze task list and existing docs to project final file structure]\"
  },
  \"dependencies\": {
    \"[project final dependencies based on task list and current dependencies]\"
  },
  \"architecture\": {
    \"main_flow\": \"[extract target flow from task list and existing docs]\",
    \"data_flow\": \"[project target data flow from architecture]\",
    \"configuration\": \"[analyze existing config approach and project improvements]\",
    \"key_components\": \"[identify target components from task analysis]\",
    \"integration_points\": \"[project target integrations from task list]\"
  },
  \"development\": {
    \"approach\": \"manifest-driven development retrofitted to existing codebase\",
    \"target_phase\": \"all_tasks_completed\",
    \"completion_criteria\": [
      \"All tasks in task_list.md completed\",
      \"All acceptance criteria met\",
      \"Production deployment ready\",
      \"Documentation complete\"
    ]
  }
}
```

### 7. Create Manifest Evolution Log
Create `docs/manifest_evolution.md` to track architectural changes:

```markdown
# Manifest Evolution Log

This document tracks changes to the proposed final manifest as the project evolves through manifest-driven development.

## Initial Version - [Current Date]

### Source
Created during project retrofit for manifest-driven development from existing codebase analysis and task planning.

### Current State
- Existing codebase analyzed and documented in codebase_manifest.json
- Task list converted to manifest-driven format
- [X] tasks with [Y] subtasks identified for completion

### Target Architecture
- [Key target components from task analysis]
- [Major architectural goals from task list]
- [Integration points planned]

### Development Approach
- Retrofitted existing project for manifest-driven development
- Task-by-task implementation with manifest validation
- Continuous architectural refinement based on implementation learnings

### Future Updates
Updates will be logged here as tasks are completed and architectural insights are gained.
```  
**If .gitignore exists:**
- Add manifest-driven specific entries if not present:
```
# Manifest-driven development
tasks/prepared/  # Temporary task files with full context
tasks/validation/  # Validation reports
!tasks/prepared/.gitkeep
!tasks/validation/.gitkeep
```

**If .gitignore doesn't exist:**
- Create basic .gitignore appropriate for the detected project type

### 7. Update Existing README.md (Optional)
**DO NOT replace existing README.md**

Instead, suggest adding a section about the new development workflow:
```markdown
## Development Workflow

This project uses manifest-driven development for structured task implementation:

- See \`claude.md\` for AI workflow instructions
- See \`tasks/task_list.md\` for current development tasks  
- See \`codebase_manifest.json\` for current project state

### Quick Development Commands
\`\`\`bash
# Generate/update project manifest
claude-code \"generate_manifest\"

# Complete task workflow
claude-code \"orchestrate_task Task-X.X\"
\`\`\`
```

### 8. Create .gitkeep Files
Add `.gitkeep` files to maintain directory structure:
- `tasks/prepared/.gitkeep`
- `tasks/completed/.gitkeep` 
- `tasks/validation/.gitkeep`

### 11. Verify Existing Task List
**If tasks/task_list.md exists:**
- Validate format is compatible with manifest-driven commands
- Ensure task IDs follow Task-X.X format

**If tasks/task_list.md missing:**
- Note that task list needs to be created
- Suggest analyzing existing documentation to create task list

### 12. Final Verification
- Ensure all new files are created without overwriting existing ones
- Verify manifest-driven structure is complete
- Test that `generate_manifest` command can read the existing codebase
- Confirm git repository is not disrupted

### 13. Create Commit (Optional)
**Only if requested by user:**
```bash
git add .claude/ tasks/ claude.md codebase_manifest.json docs/proposed_final_manifest.json docs/manifest_evolution.md
git commit -m \"Add manifest-driven development workflow to existing project\"
```

## Key Principles for Retrofitting:

1. **Preserve Existing Structure** - Don't move or rename existing files
2. **Additive Only** - Only add new files/directories, don't replace
3. **Analyze Don't Assume** - Extract project info from existing code/docs
4. **Respect Git History** - Work with existing repository
5. **Document Integration** - Explain how new workflow fits with existing project

## Expected Results:

After retrofitting:
- Existing project functionality unchanged
- Manifest-driven development workflow available
- All commands in `.claude/commands/` ready to use
- `codebase_manifest.json` reflects current project state
- `docs/proposed_final_manifest.json` shows target architecture
- `docs/manifest_evolution.md` ready to track changes
- `claude.md` provides clear instructions for AI development
- Ready to start task-based development using `tasks/task_list.md`

The project will be ready for manifest-driven development while preserving all existing work and structure."
```