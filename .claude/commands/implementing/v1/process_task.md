# Process Task Command

```bash
claude-code "Process and prepare a task for implementation using manifest-driven development.

## Task: Process Task for Implementation

**Task ID:** [TASK_ID - e.g., Task-1.1]

Start your response with: "🔄 **PROCESS_TASK EXECUTING** - Processing [TASK_ID] for implementation"

## Process Steps:

### 1. Read Current State
- Load current `codebase_manifest.json`
- Read task details from `tasks/task_list.md` for the specified task ID
- Understand current project architecture and dependencies

### 2. Analyze Task Requirements
- Break down what the task requires:
  - New files to create
  - Existing files to modify
  - New functions/classes to implement
  - Dependencies to add
  - Side effects to consider

### 3. Generate Expected Post-Task Manifest
Create a detailed expected manifest showing:
- New files that will be added with their purpose and exports
- Modified files with updated exports
- New dependencies that will be required
- Updated architecture information
- Any new side effects

### 4. Create Implementation Context
Generate comprehensive implementation notes:
- Step-by-step implementation approach
- Key technical decisions
- Integration points with existing code
- Testing requirements
- Acceptance criteria

### 5. Create Task File
Save complete task context as `tasks/prepared/[TASK_ID].json`:

```json
{
  \"task_id\": \"[TASK_ID]\",
  \"task_description\": \"[Full task description from task list]\",
  \"current_manifest\": { [Complete current manifest] },
  \"expected_manifest\": { [Complete expected post-task manifest] },
  \"implementation_notes\": {
    \"approach\": \"[Step-by-step implementation approach]\",
    \"files_to_create\": [
      {
        \"file\": \"path/to/file.ts\",
        \"purpose\": \"What this file does\",
        \"key_exports\": [\"list of main exports\"]
      }
    ],
    \"files_to_modify\": [
      {
        \"file\": \"path/to/existing.ts\",
        \"changes\": \"Description of changes needed\"
      }
    ],
    \"dependencies\": [\"new dependencies to add\"],
    \"integration_points\": [\"how this connects to existing code\"],
    \"testing_approach\": \"How to test this implementation\"
  },
  \"acceptance_criteria\": [
    \"Specific criteria for task completion\"
  ],
  \"estimated_complexity\": \"[Low/Medium/High]\",
  \"prerequisites\": [\"Any tasks that must be completed first\"]
}
```

### 6. Validation
- Verify the expected manifest is realistic and achievable
- Check for potential conflicts with existing code
- Ensure all dependencies are accounted for
- Validate that acceptance criteria are measurable

### 7. Output Summary
Report what was prepared:
- Task ID and description
- Number of files to create/modify
- Key dependencies and integration points
- Implementation complexity assessment
- Location of prepared task file

## Success Criteria:
- Task file created in `tasks/prepared/[TASK_ID].json`
- Expected manifest is detailed and realistic
- Implementation approach is clear and actionable
- All context needed for implementation is included

The task is now ready for implementation using the `implement_task` command."
```