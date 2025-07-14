# Commit Task Command

```bash
claude-code "Commit completed task implementation with proper git history.

## Task: Commit Task Implementation

**Task ID:** [TASK_ID - e.g., Task-1.1]

Start your response with: "💾 **COMMIT_TASK EXECUTING** - Committing [TASK_ID] implementation"

## Commit Process:

### 1. Verify Task Completion
- Confirm task has passed check_task validation
- Ensure all files are in working state
- Verify no uncommitted debugging code remains

### 2. Stage Changes
- Add all new files created during task
- Add all modified files
- Review changes to ensure they're all related to the task

### 3. Create Descriptive Commit Message
Format: `[TASK_ID]: [Brief description]`

Examples:
- `Task-1.1: Initialize project structure and configuration system`
- `Task-2.1: Implement database service with connection management`
- `Task-3.2: Add content classification with AI processing`

### 4. Include Detailed Commit Body
```
[TASK_ID]: [Brief description]

- [Specific change 1]
- [Specific change 2]
- [Specific change 3]

Files changed:
- [file1.ts] - [what changed]
- [file2.ts] - [what changed]

Manifest updated: [Yes/No]
Tests added: [Yes/No]
Breaking changes: [Yes/No]
```

### 5. Execute Commit
```bash
git add .
git commit -m \"[TASK_ID]: [Brief description]\" -m \"[Detailed body]\"
```

### 6. Update Task Status
- Mark task as completed and committed
- Update task tracking with commit hash
- Note any lessons learned

## Success Criteria:
- Clean commit with descriptive message
- All task-related changes included
- No unrelated changes included
- Task marked as completed
- Commit hash recorded

This creates a clear development history tied to task completion."
```